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


    # === Trilinear interpolation / deposition operator ===

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
            # Weighted average of 8 surrounding voxels
            values = jnp.array([
                grid[x0, y0, z0], grid[x0, y0, z1],
                grid[x0, y1, z0], grid[x0, y1, z1],
                grid[x1, y0, z0], grid[x1, y0, z1],
                grid[x1, y1, z0], grid[x1, y1, z1]
            ])
            return jnp.sum(weights * values)

        elif mode == "deposit":
             # Add weighted contribution to 8 surrounding voxels
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
            x, y, z, I, tau, J = state
            j_val = trilinear_op(j_map, x, y, z, mode="interp")
            #kappa_val = trilinear_op(kappa_map, x, y, z, mode="interp") # this have problems, solution here in comment:
            ix = jnp.clip(jnp.floor(x).astype(int), 0, Nx - 1)
            iy = jnp.clip(jnp.floor(y).astype(int), 0, Ny - 1)
            iz = jnp.clip(jnp.floor(z).astype(int), 0, Nz - 1)

            kappa_val = kappa_map[ix,iy,iz]

            ds = step_size
            d_tau = kappa_val * ds
            dI = j_val * jnp.exp(-tau) * ds
            I_new = I * jnp.exp(-d_tau) + dI
            tau_new = tau + d_tau
            J = trilinear_op(J, x, y, z, value=I_new, mode="deposit")
            x_new = x + direction[0] * ds
            y_new = y + direction[1] * ds
            z_new = z + direction[2] * ds
            return (x_new, y_new, z_new, I_new, tau_new, J)

        x0, y0, z0 = source_pos
        initial = (x0, y0, z0, 0.0, 0.0, jnp.zeros_like(j_map))
        _, _, _, _, _, J_ray = jax.lax.fori_loop(0, max_steps, body_fn, initial)
        return J_ray
    '''
    if use_pmap:
        n_devices = jax.device_count()
        if num_rays % n_devices != 0:
            raise ValueError(f"num_rays ({num_rays}) must be divisible by number of devices ({n_devices}) for pmap.")

        rays_per_device = num_rays // n_devices
        directions_sharded = directions.reshape((n_devices, rays_per_device, 3))

        @partial(jax.pmap, axis_name='devices')
        def trace_ray_batch(dir_batch):
            return jax.vmap(trace_single_ray)(dir_batch)  # [rays_per_device, Nx, Ny, Nz]

        J_all_sharded = trace_ray_batch(directions_sharded)  # [n_devices, rays_per_device, Nx, Ny, Nz]
        J_all = J_all_sharded.reshape((num_rays, *j_map.shape))  # [num_rays, Nx, Ny, Nz]
    '''
    if use_sharding:
        # Build mesh and sharding
        devices = jax.devices()
        n_devices = len(devices)

        if num_rays % n_devices != 0:
            raise ValueError(f"num_rays ({num_rays}) must be divisible by number of devices ({n_devices}).")

        mesh_shape = (n_devices,)
        mesh = jax.sharding.Mesh(np.array(devices).reshape(mesh_shape), axis_names=('x',))
        sharding = NamedSharding(mesh, P('x', None))  # rays over 'x', each ray is a (3,) vector

        # Reshape and shard directions
        directions_reshaped = directions.reshape((n_devices, -1, 3))
        directions_sharded = jax.device_put(directions_reshaped, sharding)

        # Apply vmap over each shard
        @jax.vmap  # Automatically parallel within each device
        def trace_ray_batch(dir_batch):
            return jax.vmap(trace_single_ray)(dir_batch)

        with mesh:
            J_all_sharded = trace_ray_batch(directions_sharded)  # [n_devices, rays_per_device, Nx, Ny, Nz]

        # Merge all rays
        J_all = J_all_sharded.reshape((num_rays, *j_map.shape))  # [num_rays, Nx, Ny, Nz]
    else:
        # No sharding: simple vmap over all rays
        J_all = jax.vmap(trace_single_ray)(directions)

    # Sum over all rays
    return jnp.sum(J_all, axis=0) * (4 * jnp.pi / num_rays)


def compute_radiation_field_from_multiple_sources(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False
):
    

    """
    Computes the total radiation field from multiple sources in 3D using ray tracing.

    Parameters:
        j_map (3D array): emissivity map
        kappa_map (3D array): absorption coefficient map
        source_positions (array of shape [N_sources, 3]): list of 3D coordinates of sources
        num_rays (int): number of rays per source
        step_size (float): ray marching step size
        max_steps (int): number of steps per ray
        use_sharding (bool): whether to use multi-GPU parallelization

    Returns:
        3D array: total radiation field
    """



    
    J_total = jnp.zeros_like(j_map)
    for source_pos in source_positions:
        J_total += compute_radiation_field_from_source(
                j_map,
                kappa_map,
                source_pos=jnp.array(source_pos),
                num_rays=num_rays,
                step_size=step_size,
                max_steps=max_steps,
                use_sharding = use_sharding

            )

    return J_total


