import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec as P

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
    Nx, Ny, Nz = grid.shape
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    z0 = jnp.floor(z).astype(jnp.int32)

    x1 = jnp.clip(x0 + 1, 0, Nx - 1)
    y1 = jnp.clip(y0 + 1, 0, Ny - 1)
    z1 = jnp.clip(z0 + 1, 0, Nz - 1)
    x0 = jnp.clip(x0, 0, Nx - 1)
    y0 = jnp.clip(y0, 0, Ny - 1)  # <-- fixed
    z0 = jnp.clip(z0, 0, Nz - 1)

    dx, dy, dz = (x - x0), (y - y0), (z - z0)

    w000 = (1-dx)*(1-dy)*(1-dz); w001 = (1-dx)*(1-dy)*dz
    w010 = (1-dx)*dy*(1-dz);     w011 = (1-dx)*dy*dz
    w100 = dx*(1-dy)*(1-dz);     w101 = dx*(1-dy)*dz
    w110 = dx*dy*(1-dz);         w111 = dx*dy*dz

    if mode == "interp":
        v000 = grid[x0, y0, z0]; v001 = grid[x0, y0, z1]
        v010 = grid[x0, y1, z0]; v011 = grid[x0, y1, z1]
        v100 = grid[x1, y0, z0]; v101 = grid[x1, y0, z1]
        v110 = grid[x1, y1, z0]; v111 = grid[x1, y1, z1]
        return (w000*v000 + w001*v001 + w010*v010 + w011*v011 +
                w100*v100 + w101*v101 + w110*v110 + w111*v111)

    elif mode == "deposit":
        assert value is not None
        out = grid
        out = out.at[x0, y0, z0].add(w000 * value)
        out = out.at[x0, y0, z1].add(w001 * value)
        out = out.at[x0, y1, z0].add(w010 * value)
        out = out.at[x0, y1, z1].add(w011 * value)
        out = out.at[x1, y0, z0].add(w100 * value)
        out = out.at[x1, y0, z1].add(w101 * value)
        out = out.at[x1, y1, z0].add(w110 * value)
        out = out.at[x1, y1, z1].add(w111 * value)
        return out

    else:
        raise ValueError("mode must be 'interp' or 'deposit'")


@partial(
    jax.jit,
    static_argnames=[
        "num_rays", "step_size", "use_sharding", "max_steps",
        "radiation_velocity", "ray_batch_size"  
    ]
)
def compute_radiation_field_from_source_with_time_step(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5,
    max_steps=None,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False,
    ray_batch_size=None,   
):
    """
    One time-horizon update from a single source; rays are micro-batched to limit memory.
    Returns a grid (Nx, Ny, Nz).
    """
    Nx, Ny, Nz = j_map.shape

    # Directions: keep (num_rays, 3) and unit norm
    directions = sample_sphere(num_rays).astype(jnp.float32)
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)

    # Steps along the ray
    if max_steps is None:
        max_distance = jnp.asarray(radiation_velocity) * jnp.asarray(time_step)
        max_steps = jnp.ceil(max_distance / jnp.asarray(step_size)).astype(jnp.int32)


    def trace_single_ray(direction):
        x, y, z = source_pos
        I = 0.0
        tau = 0.0
        J = jnp.zeros_like(j_map)

        def body_fn(i, state):
            x, y, z, I, tau, J = state
            j_val = trilinear_op(j_map, x, y, z, mode="interp")

            ix = jnp.clip(jnp.floor(x).astype(int), 0, Nx - 1)
            iy = jnp.clip(jnp.floor(y).astype(int), 0, Ny - 1)
            iz = jnp.clip(jnp.floor(z).astype(int), 0, Nz - 1)
            kappa_val = kappa_map[ix, iy, iz]

            d_tau = kappa_val * step_size
            dI = j_val * jnp.exp(-tau) * step_size
            I_new = I * jnp.exp(-d_tau) + dI
            tau_new = tau + d_tau

            J = trilinear_op(J, x, y, z, value=I_new, mode="deposit")

            x_new = x + direction[0] * step_size
            y_new = y + direction[1] * step_size
            z_new = z + direction[2] * step_size
            return (x_new, y_new, z_new, I_new, tau_new, J)

        state_init = (x, y, z, I, tau, J)
        final_state = jax.lax.fori_loop(0, max_steps, body_fn, state_init)
        return final_state[-1]  # J grid

    def sum_ray_batch(dir_batch):
        # dir_batch: (B, 3) -> sum of B ray grids
        return jnp.sum(jax.vmap(trace_single_ray, in_axes=(0,))(dir_batch), axis=0)

    # ---------------------------
    # Single-device micro-batch
    # ---------------------------
    if not use_sharding:
        if (ray_batch_size is None) or (ray_batch_size >= num_rays):
            J_sum = sum_ray_batch(directions)
        else:
            batch = int(ray_batch_size)                 # static
            n_full = num_rays // batch                  # static
            rem    = num_rays % batch                   # static

            # Full batches (size = batch, static)
            J_acc = jnp.zeros_like(j_map)
            def body_full(i, acc):
                start = i * batch
                # slice_size is static (= batch)
                dir_full = jax.lax.dynamic_slice_in_dim(directions, start, batch, axis=0)
                return acc + sum_ray_batch(dir_full)
            J_acc = jax.lax.fori_loop(0, n_full, body_full, J_acc)

            # Remainder (size = rem, also static because derived from static args)
            if rem:
                start_tail = n_full * batch
                dir_tail = jax.lax.dynamic_slice_in_dim(directions, start_tail, rem, axis=0)
                J_acc = J_acc + sum_ray_batch(dir_tail)

            J_sum = J_acc

        return J_sum * (4.0 * jnp.pi / num_rays)

    # ---------------------------
    # Sharded micro-batch (Mesh)
    # ---------------------------
    devices = jax.devices()
    n_devices = len(devices)
    if num_rays % n_devices != 0:
        raise ValueError(
            f"num_rays ({num_rays}) must be divisible by number of devices ({n_devices}) "
            f"when use_sharding=True."
        )
    rays_per_device = num_rays // n_devices

    # Split directions per device using static-size slices
    dir_chunks = [
        jax.lax.dynamic_slice_in_dim(directions, i * rays_per_device, rays_per_device, axis=0)
        for i in range(n_devices)
    ]
    directions_reshaped = jnp.stack(dir_chunks, axis=0)  # (n_devices, rays_per_device, 3)

    mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
    sharding = NamedSharding(mesh, P('x'))
    directions_sharded = jax.device_put(directions_reshaped, sharding)

    # Per-device reduction with micro-batching (static sizes)
    if (ray_batch_size is None) or (ray_batch_size >= rays_per_device):
        per_dev_batch = rays_per_device
    else:
        if rays_per_device % ray_batch_size == 0:
            per_dev_batch = int(ray_batch_size)
        else:
            # allow a remainder on each device, handled explicitly
            per_dev_batch = int(ray_batch_size)

    def reduce_device(dir_dev):
        # dir_dev: (rays_per_device, 3)
        if per_dev_batch == rays_per_device:
            return sum_ray_batch(dir_dev)
        else:
            nb  = rays_per_device // per_dev_batch     # static
            rem = rays_per_device %  per_dev_batch     # static
            J_acc = jnp.zeros_like(j_map)

            def body_full(j, acc):
                start = j * per_dev_batch
                db = jax.lax.dynamic_slice_in_dim(dir_dev, start, per_dev_batch, axis=0)
                return acc + sum_ray_batch(db)
            J_acc = jax.lax.fori_loop(0, nb, body_full, J_acc)

            if rem:
                start_tail = nb * per_dev_batch
                db_tail = jax.lax.dynamic_slice_in_dim(dir_dev, start_tail, rem, axis=0)
                J_acc = J_acc + sum_ray_batch(db_tail)

            return J_acc

    with mesh:
        J_per_device = jax.vmap(reduce_device, in_axes=0)(directions_sharded)  # (n_devices, Nx,Ny,Nz)

    J_sum = jnp.sum(J_per_device, axis=0)
    return J_sum * (4.0 * jnp.pi / num_rays)


def compute_radiation_field_from_multiple_sources_with_time_step(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False, unroll = False,
    ray_batch_size=None
):
    """
    Compute the radiation field within one time step from multiple sources using lax.fori_loop.
    """

    def body_fn(i, I_total):
        source_pos = source_positions[i]
        I_new = compute_radiation_field_from_source_with_time_step(
            j_map, kappa_map, source_pos,
            num_rays=num_rays,
            step_size=step_size,
            radiation_velocity=radiation_velocity,
            time_step=time_step,
            use_sharding=use_sharding,
            ray_batch_size=ray_batch_size
        )
        return I_total + I_new

    num_sources = source_positions.shape[0]
    I_total = jnp.zeros_like(j_map)
    I_total = jax.lax.fori_loop(0, num_sources, body_fn, I_total, unroll = unroll)

    return I_total
