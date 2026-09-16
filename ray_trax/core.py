"""
core.py — shared ray-marching primitives for every Ray-trax solver.

All solvers (steady-state, time-dependent, multi-frequency, direction-resolved)
are thin wrappers around three building blocks defined here:

  sample_sphere   Fibonacci / golden-angle lattice of unit directions on S².
  trace_ray       March one ray from a source over a fixed number of steps,
                  depositing the attenuated intensity onto the grid.
  sum_over_rays   Sum a per-ray tracer over all directions, with optional ray
                  micro-batching (peak-memory cap) and multi-device sharding.

Grids are (Nx, Ny, Nz, *trailing). A scalar solver has no trailing axis; the
multi-frequency solver carries a trailing frequency axis. Interpolation weights
depend only on position, so all trailing components share the same weights and
are handled by broadcasting.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------

def sample_sphere(n):
    """Nearly uniform unit directions on S² via the Fibonacci lattice -> (n, 3)."""
    i = jnp.arange(n)
    golden_ratio = (1 + jnp.sqrt(5)) / 2
    phi = 2 * jnp.pi * i / golden_ratio
    cos_theta = 1 - 2 * (i + 0.5) / n
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    return jnp.stack([sin_theta * jnp.cos(phi),
                      sin_theta * jnp.sin(phi),
                      cos_theta], axis=1)


# ---------------------------------------------------------------------------
# Grid access
# ---------------------------------------------------------------------------

def _corners(shape3, x, y, z):
    """Eight corner indices (clipped to the grid) and trilinear weights for (x, y, z)."""
    Nx, Ny, Nz = shape3
    fx, fy, fz = jnp.floor(x), jnp.floor(y), jnp.floor(z)
    x0 = jnp.clip(fx.astype(jnp.int32), 0, Nx - 1)
    y0 = jnp.clip(fy.astype(jnp.int32), 0, Ny - 1)
    z0 = jnp.clip(fz.astype(jnp.int32), 0, Nz - 1)
    x1 = jnp.clip(x0 + 1, 0, Nx - 1)
    y1 = jnp.clip(y0 + 1, 0, Ny - 1)
    z1 = jnp.clip(z0 + 1, 0, Nz - 1)

    # fractional offsets, always in [0, 1) regardless of the clipping above
    dx, dy, dz = x - fx, y - fy, z - fz

    weights = (
        (1 - dx) * (1 - dy) * (1 - dz), (1 - dx) * (1 - dy) * dz,
        (1 - dx) * dy * (1 - dz),       (1 - dx) * dy * dz,
        dx * (1 - dy) * (1 - dz),       dx * (1 - dy) * dz,
        dx * dy * (1 - dz),             dx * dy * dz,
    )
    corners = (
        (x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1),
        (x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1),
    )
    return weights, corners


def trilinear_op(grid, x, y, z, value=None, mode="interp"):
    """Trilinear interpolation from, or symmetric deposition onto, `grid`.

    grid  : (Nx, Ny, Nz, *trailing)
    mode  : "interp"  -> returns grid value at (x, y, z), shape `trailing`
            "deposit" -> returns grid with `value` (shape `trailing`) spread over
                         the 8 surrounding voxels with trilinear weights
    Out-of-range coordinates are clipped to the nearest voxel; callers that
    march rays gate deposition with `inside_domain` so nothing piles up on faces.
    """
    weights, corners = _corners(grid.shape[:3], x, y, z)

    if mode == "interp":
        return sum(w * grid[ix, iy, iz] for w, (ix, iy, iz) in zip(weights, corners))

    if mode == "deposit":
        assert value is not None, "value is required for mode='deposit'"
        out = grid
        for w, (ix, iy, iz) in zip(weights, corners):
            out = out.at[ix, iy, iz].add(w * value)
        return out

    raise ValueError("mode must be 'interp' or 'deposit'")


def nearest_voxel(grid, x, y, z):
    """Nearest-voxel lookup (floor + clip); piecewise constant, no gradient w.r.t. position."""
    Nx, Ny, Nz = grid.shape[:3]
    ix = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, Nx - 1)
    iy = jnp.clip(jnp.floor(y).astype(jnp.int32), 0, Ny - 1)
    iz = jnp.clip(jnp.floor(z).astype(jnp.int32), 0, Nz - 1)
    return grid[ix, iy, iz]


def inside_domain(x, y, z, shape3, dtype):
    """1 where (x, y, z) lies within the closed voxel range [0, N-1]^3, else 0.

    Matches the clipping in `trilinear_op`, so both faces of every axis are
    treated symmetrically. Depends only on ray geometry, hence carries no
    gradient w.r.t. the fields.
    """
    Nx, Ny, Nz = shape3
    return ((x >= 0) & (x <= Nx - 1) &
            (y >= 0) & (y <= Ny - 1) &
            (z >= 0) & (z <= Nz - 1)).astype(dtype)


# ---------------------------------------------------------------------------
# Single-ray integrator
# ---------------------------------------------------------------------------

def trace_ray(j_map, kappa_map, source_pos, direction, *,
              step_size, max_steps, kappa_interp="nearest", with_direction=False):
    """March one ray and return the intensity it deposits on the grid.

    Semi-analytic update per step of length ds:
        I_new = I·exp(−κ ds) + j·exp(−τ)·ds ,   τ_new = τ + κ ds
    with j from trilinear interpolation and κ from a nearest-voxel lookup
    (or trilinear if kappa_interp="trilinear", which is differentiable w.r.t.
    kappa_map). I_new is deposited trilinearly at the current position while
    the ray is inside the domain.

    Parameters
    ----------
    j_map, kappa_map : (Nx, Ny, Nz, *trailing)
    source_pos       : (3,) grid coordinates (int or float)
    direction        : (3,) unit vector
    step_size        : float, ray step in grid units (static)
    max_steps        : int, number of steps (static for reverse-mode autodiff)
    kappa_interp     : "nearest" | "trilinear"
    with_direction   : also accumulate D = Σ I·n̂ (shape (Nx, Ny, Nz, 3));
                       only for scalar (no trailing axis) grids

    Returns
    -------
    J            : (Nx, Ny, Nz, *trailing)          if not with_direction
    (J, D)       : J as above, D (Nx, Ny, Nz, 3)     if with_direction
    """
    shape3 = j_map.shape[:3]
    trailing = j_map.shape[3:]
    dtype = j_map.dtype
    ds = step_size

    if kappa_interp == "trilinear":
        kappa_at = partial(trilinear_op, kappa_map, mode="interp")
    else:
        kappa_at = partial(nearest_voxel, kappa_map)

    def body_fn(_, state):
        x, y, z, I, tau, J, D = state

        j_val = trilinear_op(j_map, x, y, z, mode="interp")
        kappa_val = kappa_at(x, y, z)

        d_tau = kappa_val * ds
        I_new = I * jnp.exp(-d_tau) + j_val * jnp.exp(-tau) * ds
        tau_new = tau + d_tau

        # Deposit only while inside the domain; otherwise the clipping in
        # trilinear_op would funnel every out-of-box step onto the faces.
        I_dep = I_new * inside_domain(x, y, z, shape3, dtype)
        J = trilinear_op(J, x, y, z, value=I_dep, mode="deposit")
        if with_direction:
            D = trilinear_op(D, x, y, z, value=I_dep * direction, mode="deposit")

        return (x + direction[0] * ds,
                y + direction[1] * ds,
                z + direction[2] * ds,
                I_new, tau_new, J, D)

    # Float carry even for integer voxel tuples (fori_loop requires a fixed carry dtype).
    x0, y0, z0 = jnp.asarray(source_pos, dtype)
    # jnp.zeros(shape, dtype) rather than zeros_like: inside a shard_map body,
    # zeros_like would inherit the closed-over array's Auto-mesh sharding.
    init = (x0, y0, z0,
            jnp.zeros(trailing, dtype), jnp.zeros(trailing, dtype),
            jnp.zeros(j_map.shape, dtype),
            jnp.zeros((*shape3, 3), dtype) if with_direction else None)

    *_, J, D = jax.lax.fori_loop(0, max_steps, body_fn, init)
    return (J, D) if with_direction else J


# ---------------------------------------------------------------------------
# Reduction over rays: micro-batching + optional multi-device sharding
# ---------------------------------------------------------------------------

def sum_over_rays(trace_fn, directions, *, num_rays, use_sharding=False, ray_batch_size=None):
    """Σ_rays trace_fn(direction), for `directions` of shape (num_rays, 3).

    trace_fn may return any pytree of arrays (e.g. J, or (J, D)); the sum is
    taken leaf-wise over the ray axis.

    ray_batch_size : int | None  micro-batch rays through a fori_loop to cap peak
                     memory (None = vmap over all rays at once).
    use_sharding   : shard rays across all visible devices with shard_map; each
                     device sums its chunk and the partial sums are all-reduced.
                     num_rays must be divisible by the device count.
    """
    tree = jax.tree_util

    def sum_batch(dir_batch):
        return tree.tree_map(lambda a: jnp.sum(a, axis=0), jax.vmap(trace_fn)(dir_batch))

    def add(a, b):
        return tree.tree_map(jnp.add, a, b)

    def zeros():
        # Built from the abstract output so no sharding is inherited from inputs.
        return tree.tree_map(lambda s: jnp.zeros(s.shape, s.dtype),
                             jax.eval_shape(trace_fn, directions[0]))

    def batched_sum(dir_all, n_rays):
        if (ray_batch_size is None) or (ray_batch_size >= n_rays):
            return sum_batch(dir_all)
        batch = int(ray_batch_size)
        n_full, rem = divmod(n_rays, batch)

        def body(i, acc):
            db = jax.lax.dynamic_slice_in_dim(dir_all, i * batch, batch, axis=0)
            return add(acc, sum_batch(db))

        acc = jax.lax.fori_loop(0, n_full, body, zeros())
        if rem:
            db_tail = jax.lax.dynamic_slice_in_dim(dir_all, n_full * batch, rem, axis=0)
            acc = add(acc, sum_batch(db_tail))
        return acc

    if not use_sharding:
        return batched_sum(directions, num_rays)

    n_devices = len(jax.devices())
    if num_rays % n_devices != 0:
        raise ValueError(
            f"num_rays ({num_rays}) must be divisible by n_devices ({n_devices}) "
            f"when use_sharding=True."
        )
    rays_per_device = num_rays // n_devices
    mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=("x",))

    def per_device_sum(dir_chunk):
        local = batched_sum(dir_chunk, rays_per_device)
        return tree.tree_map(lambda a: jax.lax.psum(a, axis_name="x"), local)

    return shard_map(
        per_device_sum,
        mesh=mesh,
        in_specs=(P("x", None),),   # rays sharded across devices
        out_specs=P(),               # replicated after psum
        check_rep=False,             # closed-over fields are implicitly replicated
    )(directions)
