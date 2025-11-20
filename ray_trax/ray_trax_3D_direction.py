from functools import partial
import jax
import jax.numpy as jnp
from ray_trax.utils import sample_sphere, trilinear_op
# ------------------------------------------------------------
# Time-dependent, micro-batched, optional sharded ray tracing
# Returns: (J_total, direction_map)
#   - J_total: (Nx, Ny, Nz)
#   - direction_map: (Nx, Ny, Nz, 3) = (sum I * n) / (sum I + eps)
# ------------------------------------------------------------
def deposit_vector(D, x, y, z, vec, val):
    # D: (Nx,Ny,Nz,3), vec: (3,), val: scalar
    # deposits (val * vec[k]) into each D[...,k] using scalar trilinear_op
    Dx = trilinear_op(D[..., 0], x, y, z, value=val * vec[0], mode="deposit")
    Dy = trilinear_op(D[..., 1], x, y, z, value=val * vec[1], mode="deposit")
    Dz = trilinear_op(D[..., 2], x, y, z, value=val * vec[2], mode="deposit")
    return jnp.stack([Dx, Dy, Dz], axis=-1)


@partial(
    jax.jit,
    static_argnames=(
        "num_rays",
        "step_size",
        "use_sharding",
        "max_steps",
        "radiation_velocity",
        "ray_batch_size",
    ),
)
def compute_radiation_field_from_source_with_time_step_direction(
    j_map,
    kappa_map,
    source_pos,
    *,
    num_rays: int = 4096,
    step_size: float = 0.5,
    max_steps=None,                  # optional: override number of substeps
    radiation_velocity: float = 1.0, # "c" in code units
    time_step: float = 1.0,          # horizon: c * time_step
    use_sharding: bool = False,
    ray_batch_size: int | None = None,
    eps: float = 1e-12,
    directions: jnp.ndarray | None = None,  # optional (num_rays, 3)
):
    """
    Time-dependent tracing from a single source over a horizon c * time_step,
    using micro-batches of rays to reduce peak memory. Returns both the total
    scalar intensity J and a per-voxel direction map (sum I*n / sum I).
    """
    Nx, Ny, Nz = j_map.shape

    # --- directions (keep unit norm; reuse provided or sample once) ---
    if directions is None:
        dirs = sample_sphere(num_rays).astype(jnp.float32)  # (num_rays, 3)
    else:
        dirs = jnp.asarray(directions, dtype=jnp.float32)
        if dirs.shape != (num_rays, 3):
            raise ValueError(f"directions must have shape ({num_rays}, 3), got {dirs.shape}")
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

    # --- steps along the ray: derive from physics if not provided ---
    if max_steps is None:
        max_distance = jnp.asarray(radiation_velocity) * jnp.asarray(time_step)
        max_steps = jnp.ceil(max_distance / jnp.asarray(step_size)).astype(jnp.int32)

    # --- per-ray tracer (returns scalar J grid and vector D grid) ---
    def trace_single_ray(direction):
        x, y, z = source_pos  # assumed voxel coords
        I = 0.0
        tau = 0.0
        J = jnp.zeros_like(j_map)                 # scalar accumulator
        D = jnp.zeros((*j_map.shape, 3), J.dtype) # vector accumulator

        def body_fn(i, state):
            x, y, z, I, tau, J, D = state

            # emissivity via trilinear interpolation
            j_val = trilinear_op(j_map, x, y, z, mode="interp")

            # kappa via nearest-neighbor (as in your original)
            ix = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, Nx - 1)
            iy = jnp.clip(jnp.floor(y).astype(jnp.int32), 0, Ny - 1)
            iz = jnp.clip(jnp.floor(z).astype(jnp.int32), 0, Nz - 1)  # <- z, not y
            kappa_val = kappa_map[ix, iy, iz]

            d_tau = kappa_val * step_size
            dI = j_val * jnp.exp(-tau) * step_size
            I_new = I * jnp.exp(-d_tau) + dI
            tau_new = tau + d_tau

            # deposit: same weights for J and D (D gets direction weighting)
            J = trilinear_op(J, x, y, z, value=I_new, mode="deposit")
            D = deposit_vector(D, x, y, z, direction, I_new)  


            # advance
            x_new = x + direction[0] * step_size
            y_new = y + direction[1] * step_size
            z_new = z + direction[2] * step_size
            return (x_new, y_new, z_new, I_new, tau_new, J, D)

        state0 = (x, y, z, I, tau, J, D)
        xf, yf, zf, If, tauf, Jf, Df = jax.lax.fori_loop(0, max_steps, body_fn, state0)
        return Jf, Df

    # --- sum a batch of rays → (J_batch_sum, D_batch_sum) ---
    def sum_ray_batch(dir_batch):
        Jb, Db = jax.vmap(trace_single_ray, in_axes=(0,))(dir_batch)  # Jb:(B,Nx,Ny,Nz), Db:(B,Nx,Ny,Nz,3)
        return jnp.sum(Jb, axis=0), jnp.sum(Db, axis=0)

    # =========================
    # Single-device micro-batch
    # =========================
    if not use_sharding:
        if (ray_batch_size is None) or (ray_batch_size >= num_rays):
            J_sum, D_sum = sum_ray_batch(dirs)
        else:
            batch = int(ray_batch_size)
            n_full = num_rays // batch
            rem    = num_rays %  batch

            J_acc = jnp.zeros_like(j_map)
            D_acc = jnp.zeros((*j_map.shape, 3), J_acc.dtype)

            def body_full(i, acc):
                Jc, Dc = acc
                start = i * batch
                dir_full = jax.lax.dynamic_slice_in_dim(dirs, start, batch, axis=0)
                Jb, Db = sum_ray_batch(dir_full)
                return (Jc + Jb, Dc + Db)

            J_acc, D_acc = jax.lax.fori_loop(0, n_full, body_full, (J_acc, D_acc))

            if rem:
                start_tail = n_full * batch
                dir_tail = jax.lax.dynamic_slice_in_dim(dirs, start_tail, rem, axis=0)
                Jb, Db = sum_ray_batch(dir_tail)
                J_acc, D_acc = (J_acc + Jb, D_acc + Db)

            J_sum, D_sum = J_acc, D_acc

        # solid-angle weight for Monte Carlo / quasi-MC integration
        w = (4.0 * jnp.pi) / num_rays
        J_total = J_sum * w
        D_total = D_sum * w
        direction_map = D_total / (J_total[..., None] + eps)
        return J_total, direction_map

    # =========================
    # Multi-device micro-batch
    # =========================
    n_devices = jax.local_device_count()
    if n_devices <= 1:
        # fallback to single-device path above
        w = (4.0 * jnp.pi) / num_rays
        if (ray_batch_size is None) or (ray_batch_size >= num_rays):
            J_sum, D_sum = sum_ray_batch(dirs)
        else:
            batch = int(ray_batch_size)
            n_full = num_rays // batch
            rem    = num_rays %  batch
            J_acc = jnp.zeros_like(j_map)
            D_acc = jnp.zeros((*j_map.shape, 3), J_acc.dtype)
            def body_full(i, acc):
                Jc, Dc = acc
                start = i * batch
                dir_full = jax.lax.dynamic_slice_in_dim(dirs, start, batch, axis=0)
                Jb, Db = sum_ray_batch(dir_full)
                return (Jc + Jb, Dc + Db)
            J_acc, D_acc = jax.lax.fori_loop(0, n_full, body_full, (J_acc, D_acc))
            if rem:
                start_tail = n_full * batch
                dir_tail = jax.lax.dynamic_slice_in_dim(dirs, start_tail, rem, axis=0)
                Jb, Db = sum_ray_batch(dir_tail)
                J_acc, D_acc = (J_acc + Jb, D_acc + Db)
            J_sum, D_sum = J_acc, D_acc
        J_total = J_sum * w
        D_total = D_sum * w
        direction_map = D_total / (J_total[..., None] + eps)
        return J_total, direction_map

    # require equal split across devices
    if num_rays % n_devices != 0:
        raise ValueError(
            f"`num_rays` ({num_rays}) must be divisible by number of devices ({n_devices}) when use_sharding=True."
        )
    rays_per_dev = num_rays // n_devices

    # static-size splits per device (no reshape that could mix the (x,y,z) axis)
    dir_chunks = [
        jax.lax.dynamic_slice_in_dim(dirs, i * rays_per_dev, rays_per_dev, axis=0)
        for i in range(n_devices)
    ]
    dir_by_dev = jnp.stack(dir_chunks, axis=0)  # (n_devices, rays_per_dev, 3)

    # per-device micro-batch size
    if (ray_batch_size is None) or (ray_batch_size >= rays_per_dev):
        per_dev_batch = rays_per_dev
    else:
        per_dev_batch = int(ray_batch_size)

    def reduce_device(dir_dev):
        # dir_dev: (rays_per_dev, 3) on this device
        if per_dev_batch == rays_per_dev:
            return sum_ray_batch(dir_dev)
        else:
            nb  = rays_per_dev // per_dev_batch
            rem = rays_per_dev %  per_dev_batch
            J_acc = jnp.zeros_like(j_map)
            D_acc = jnp.zeros((*j_map.shape, 3), J_acc.dtype)

            def body_full(j, acc):
                Jc, Dc = acc
                start = j * per_dev_batch
                db = jax.lax.dynamic_slice_in_dim(dir_dev, start, per_dev_batch, axis=0)
                Jb, Db = sum_ray_batch(db)
                return (Jc + Jb, Dc + Db)

            J_acc, D_acc = jax.lax.fori_loop(0, nb, body_full, (J_acc, D_acc))

            if rem:
                start_tail = nb * per_dev_batch
                db_tail = jax.lax.dynamic_slice_in_dim(dir_dev, start_tail, rem, axis=0)
                Jb, Db = sum_ray_batch(db_tail)
                J_acc, D_acc = (J_acc + Jb, D_acc + Db)

            return J_acc, D_acc

    # pmap over devices, then sum-reduce
    J_per_dev, D_per_dev = jax.pmap(reduce_device, axis_name="devices")(dir_by_dev)
    J_sum = jnp.sum(J_per_dev, axis=0)
    D_sum = jnp.sum(D_per_dev, axis=0)

    w = (4.0 * jnp.pi) / num_rays
    J_total = J_sum * w
    D_total = D_sum * w
    direction_map = D_total / (J_total[..., None] + eps)
    return J_total, direction_map

