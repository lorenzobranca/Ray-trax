import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec as P

@partial(jax.jit, static_argnames=["num_rays", "step_size", "max_steps", "use_sharding"])
def compute_radiation_field_from_source(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False
):
    Nx, Ny, Nz = j_map.shape

    def sample_sphere(n):
        i = jnp.arange(0, n)
        golden_ratio = (1 + jnp.sqrt(5)) / 2
        phi = 2 * jnp.pi * i / golden_ratio
        cos_theta = 1 - 2 * (i + 0.5) / n
        sin_theta = jnp.sqrt(1 - cos_theta**2)
        x = sin_theta * jnp.cos(phi)
        y = sin_theta * jnp.sin(phi)
        z = cos_theta
        return jnp.stack([x, y, z], axis=1)

    def trilinear_op(grid, x, y, z, value=None, mode="interp"):
        x0 = jnp.floor(x).astype(int)
        y0 = jnp.floor(y).astype(int)
        z0 = jnp.floor(z).astype(int)
        x1 = jnp.clip(x0 + 1, 0, Nx - 1)
        y1 = jnp.clip(y0 + 1, 0, Ny - 1)
        z1 = jnp.clip(z0 + 1, 0, Nz - 1)
        x0 = jnp.clip(x0, 0, Nx - 1)
        y0 = jnp.clip(y0, 0, Ny - 1)
        z0 = jnp.clip(z0, 0, Nz - 1)

        dx = x - x0
        dy = y - y0
        dz = z - z0

        weights = jnp.array([
            (1 - dx) * (1 - dy) * (1 - dz),
            (1 - dx) * (1 - dy) * dz,
            (1 - dx) * dy * (1 - dz),
            (1 - dx) * dy * dz,
            dx * (1 - dy) * (1 - dz),
            dx * (1 - dy) * dz,
            dx * dy * (1 - dz),
            dx * dy * dz
        ])

        if mode == "interp":
            values = jnp.array([
                grid[x0, y0, z0], grid[x0, y0, z1],
                grid[x0, y1, z0], grid[x0, y1, z1],
                grid[x1, y0, z0], grid[x1, y0, z1],
                grid[x1, y1, z0], grid[x1, y1, z1]
            ])
            return jnp.sum(weights * values)

        elif mode == "deposit":
            assert value is not None
            updates = jnp.zeros_like(grid)
            indices = [
                (x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1),
                (x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1)
            ]
            for w, (ix, iy, iz) in zip(weights, indices):
                updates = updates.at[ix, iy, iz].add(w * value)
            return grid + updates

    directions = sample_sphere(num_rays)

    def trace_single_ray(direction):
        def body_fn(i, state):
            x, y, z, I, tau, J, D = state
            j_val = trilinear_op(j_map, x, y, z, mode="interp")
            #kappa_val = trilinear_op(kappa_map, x, y, z, mode="interp") #can have issues, see solution below
            ix = jnp.clip(jnp.floor(x).astype(int), 0, Nx - 1)
            iy = jnp.clip(jnp.floor(y).astype(int), 0, Ny - 1)
            iz = jnp.clip(jnp.floor(y).astype(int), 0, Nz - 1)

            kappa_val = kappa_map[ix,iy,iz]

            ds = step_size
            d_tau = kappa_val * ds
            dI = j_val * jnp.exp(-tau) * ds
            I_new = I * jnp.exp(-d_tau) + dI
            tau_new = tau + d_tau
            J = trilinear_op(J, x, y, z, value=I_new, mode="deposit")
            D = trilinear_op(D, x, y, z, value=I_new * direction, mode="deposit")
            x_new = x + direction[0] * ds
            y_new = y + direction[1] * ds
            z_new = z + direction[2] * ds
            return (x_new, y_new, z_new, I_new, tau_new, J, D)

        x0, y0, z0 = source_pos
        initial = (x0, y0, z0, 0.0, 0.0, jnp.zeros_like(j_map), jnp.zeros((*j_map.shape, 3)))
        _, _, _, _, _, J_ray, D_ray = jax.lax.fori_loop(0, max_steps, body_fn, initial)
        return J_ray, D_ray

    if use_sharding:
        devices = jax.devices()
        n_devices = len(devices)

        if num_rays % n_devices != 0:
            raise ValueError(f"num_rays ({num_rays}) must be divisible by number of devices ({n_devices}).")

        mesh_shape = (n_devices,)
        mesh = jax.sharding.Mesh(np.array(devices).reshape(mesh_shape), axis_names=('x',))
        sharding = NamedSharding(mesh, P('x', None))

        directions_reshaped = directions.reshape((n_devices, -1, 3))
        directions_sharded = jax.device_put(directions_reshaped, sharding)

        @jax.vmap
        def trace_ray_batch(dir_batch):
            return jax.vmap(trace_single_ray)(dir_batch)

        with mesh:
            J_all_sharded, D_all_sharded = trace_ray_batch(directions_sharded)

        J_all = J_all_sharded.reshape((num_rays, *j_map.shape))
        D_all = D_all_sharded.reshape((num_rays, *j_map.shape, 3))
    else:
        J_all, D_all = jax.vmap(trace_single_ray)(directions)

    J_total = jnp.sum(J_all, axis=0) * (4 * jnp.pi / num_rays)
    D_total = jnp.sum(D_all, axis=0) * (4 * jnp.pi / num_rays)
    direction_map = D_total / (J_total[..., None] + 1e-12)

    return J_total, direction_map

def compute_radiation_field_from_multiple_sources(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False
):
    J_total = jnp.zeros_like(j_map)
    D_total = jnp.zeros((*j_map.shape, 3))

    for source_pos in source_positions:
        J_src, D_src = compute_radiation_field_from_source(
            j_map,
            kappa_map,
            source_pos=jnp.array(source_pos),
            num_rays=num_rays,
            step_size=step_size,
            max_steps=max_steps,
            use_sharding=use_sharding
        )
        J_total += J_src
        D_total += D_src * (J_src[..., None])  # Weighted sum

    direction_map = D_total / (J_total[..., None] + 1e-12)
    return J_total, direction_map

