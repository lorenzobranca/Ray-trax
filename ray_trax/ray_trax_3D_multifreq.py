"""
3D multi-frequency ray-tracing solver for the RTE.

Drop-in extension of ray_trax_3D.py for spectral transport.
The only structural difference is that j_map, kappa_map and the output J
carry a trailing frequency axis of length n_freq.  Inside each ray step
I and tau are (n_freq,) vectors; all operations are elementwise over
frequency.  Frequency bins are independent (no scattering).

Inputs
------
j_map     : (Nx, Ny, Nz, n_freq)   emissivity  [e.g. erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹]
kappa_map : (Nx, Ny, Nz, n_freq)   opacity     [cm⁻¹]
            built with cross_sections.build_kappa_map()

Output
------
J         : (Nx, Ny, Nz, n_freq)   mean intensity, normalised by 4π / num_rays
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


@partial(jax.jit, static_argnames=["num_rays", "step_size", "max_steps",
                                    "use_sharding", "ray_batch_size", "kappa_interp"])
def compute_radiation_field_multifreq(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False, ray_batch_size=None,
    kappa_interp="nearest",
):
    """
    Mean radiation field J_ν from a single point source for all frequency bins.

    Parameters
    ----------
    j_map        : (Nx, Ny, Nz, n_freq)
    kappa_map    : (Nx, Ny, Nz, n_freq)
    source_pos   : (3,)  source position in grid coordinates
    num_rays     : int   Fibonacci-lattice directions
    step_size    : float ray step in grid units
    max_steps    : int   steps per ray
    use_sharding   : bool  multi-GPU via shard_map; num_rays must be divisible by n_devices
    ray_batch_size : int | None  micro-batch rays to cap peak memory
    kappa_interp : "nearest" (default, production) or "trilinear" (differentiable w.r.t. kappa_map)

    Returns
    -------
    J : (Nx, Ny, Nz, n_freq)
    """
    Nx, Ny, Nz, n_freq = j_map.shape

    # --- Fibonacci / golden-ratio direction lattice ---
    def sample_sphere(n):
        i = jnp.arange(n)
        golden_ratio = (1 + jnp.sqrt(5)) / 2
        phi       = 2 * jnp.pi * i / golden_ratio
        cos_theta = 1 - 2 * (i + 0.5) / n
        sin_theta = jnp.sqrt(1 - cos_theta ** 2)
        return jnp.stack([sin_theta * jnp.cos(phi),
                          sin_theta * jnp.sin(phi),
                          cos_theta], axis=1)

    # --- Trilinear interpolation / deposition over (Nx,Ny,Nz,n_freq) grids ---
    #
    # interp : grid[x,y,z,:]  → (n_freq,) weighted average over 8 corners
    # deposit: grid += w_k * value  at each of the 8 corners; value is (n_freq,)
    #
    # Weights depend only on fractional position, identical for all frequencies.

    def trilinear_op(grid, x, y, z, value=None, mode="interp"):
        x0 = jnp.clip(jnp.floor(x).astype(int), 0, Nx - 1)
        y0 = jnp.clip(jnp.floor(y).astype(int), 0, Ny - 1)
        z0 = jnp.clip(jnp.floor(z).astype(int), 0, Nz - 1)
        x1 = jnp.clip(x0 + 1, 0, Nx - 1)
        y1 = jnp.clip(y0 + 1, 0, Ny - 1)
        z1 = jnp.clip(z0 + 1, 0, Nz - 1)

        # fractional offset — always in [0,1) regardless of boundary clipping
        dx = x - jnp.floor(x)
        dy = y - jnp.floor(y)
        dz = z - jnp.floor(z)

        w = jnp.array([
            (1-dx)*(1-dy)*(1-dz),
            (1-dx)*(1-dy)*   dz,
            (1-dx)*   dy*(1-dz),
            (1-dx)*   dy*   dz,
               dx *(1-dy)*(1-dz),
               dx *(1-dy)*   dz,
               dx *   dy*(1-dz),
               dx *   dy*   dz,
        ])  # (8,)

        corners = [
            (x0,y0,z0), (x0,y0,z1), (x0,y1,z0), (x0,y1,z1),
            (x1,y0,z0), (x1,y0,z1), (x1,y1,z0), (x1,y1,z1),
        ]

        if mode == "interp":
            # grid[ix,iy,iz] → (n_freq,) ; stack to (8, n_freq)
            vals = jnp.stack([grid[ix, iy, iz] for ix, iy, iz in corners])
            return jnp.einsum('i,ij->j', w, vals)   # (n_freq,)

        else:  # deposit
            # value : (n_freq,) ; w[k] scalar → scatter weighted contribution
            # NB: jnp.zeros(shape, dtype) — NOT zeros_like — so we don't inherit the
            # closed-over array's (Auto-mesh) sharding inside the Manual shard_map body.
            updates = jnp.zeros(grid.shape, grid.dtype)
            for wk, (ix, iy, iz) in zip(w, corners):
                updates = updates.at[ix, iy, iz].add(wk * value)
            return grid + updates

    directions = sample_sphere(num_rays)

    # --- Single-ray integrator ---
    def trace_single_ray(direction):
        def body_fn(_, state):
            x, y, z, I, tau, J = state

            # emissivity via trilinear interp → (n_freq,)
            j_val = trilinear_op(j_map, x, y, z, mode="interp")

            # opacity lookup — nearest-neighbour (stable) or trilinear (differentiable)
            if kappa_interp == "trilinear":
                kappa_val = trilinear_op(kappa_map, x, y, z, mode="interp")
            else:
                ix = jnp.clip(jnp.floor(x).astype(int), 0, Nx - 1)
                iy = jnp.clip(jnp.floor(y).astype(int), 0, Ny - 1)
                iz = jnp.clip(jnp.floor(z).astype(int), 0, Nz - 1)
                kappa_val = kappa_map[ix, iy, iz]   # (n_freq,)

            ds      = step_size
            d_tau   = kappa_val * ds                              # (n_freq,)
            I_new   = I * jnp.exp(-d_tau) + j_val * jnp.exp(-tau) * ds   # (n_freq,)
            tau_new = tau + d_tau                                 # (n_freq,)

            # Only deposit while the ray is inside the domain. Without this gate,
            # coordinate clipping in trilinear_op funnels every out-of-box step
            # onto the nearest boundary voxel, piling up spurious flux on the
            # domain faces. `inside` depends only on ray geometry (not on
            # j/kappa/n_HI), so it carries no gradient and stays JIT-safe.
            inside = ((x >= 0) & (x < Nx) &
                      (y >= 0) & (y < Ny) &
                      (z >= 0) & (z < Nz)).astype(j_map.dtype)
            J_new = trilinear_op(J, x, y, z, value=I_new * inside, mode="deposit")

            return (x + direction[0] * ds,
                    y + direction[1] * ds,
                    z + direction[2] * ds,
                    I_new, tau_new, J_new)

        x0, y0, z0 = source_pos
        init = (x0, y0, z0,
                jnp.zeros(n_freq),
                jnp.zeros(n_freq),
                jnp.zeros(j_map.shape, j_map.dtype))   # not zeros_like: avoid sharding inheritance
        _, _, _, _, _, J_ray = jax.lax.fori_loop(0, max_steps, body_fn, init)
        return J_ray   # (Nx, Ny, Nz, n_freq)

    def sum_ray_batch(dir_batch):
        # vmap over rays, sum over the ray axis
        return jnp.sum(jax.vmap(trace_single_ray)(dir_batch), axis=0)

    def batched_sum(dir_all, n_rays):
        """Sum ray contributions with optional micro-batching to cap peak memory."""
        if (ray_batch_size is None) or (ray_batch_size >= n_rays):
            return sum_ray_batch(dir_all)
        batch  = int(ray_batch_size)
        n_full = n_rays // batch
        rem    = n_rays %  batch
        J_acc  = jnp.zeros(j_map.shape, j_map.dtype)   # not zeros_like: avoid sharding inheritance
        def body(i, acc):
            db = jax.lax.dynamic_slice_in_dim(dir_all, i * batch, batch, axis=0)
            return acc + sum_ray_batch(db)
        J_acc = jax.lax.fori_loop(0, n_full, body, J_acc)
        if rem:
            db_tail = jax.lax.dynamic_slice_in_dim(dir_all, n_full * batch, rem, axis=0)
            J_acc   = J_acc + sum_ray_batch(db_tail)
        return J_acc

    # --- Multi-GPU path ---
    if use_sharding:
        n_devices = len(jax.devices())
        if num_rays % n_devices != 0:
            raise ValueError(
                f"num_rays ({num_rays}) must be divisible by n_devices ({n_devices})."
            )
        rays_per_device = num_rays // n_devices
        mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))

        def per_device_sum(dir_chunk):
            J_local = batched_sum(dir_chunk, rays_per_device)
            return jax.lax.psum(J_local, axis_name='x')

        J_sum = shard_map(
            per_device_sum,
            mesh=mesh,
            in_specs=(P('x', None),),   # shard rays across devices
            out_specs=P(),               # replicated after psum
            check_rep=False,
        )(directions)
    else:
        J_sum = batched_sum(directions, num_rays)

    return J_sum * (4 * jnp.pi / num_rays)


def compute_radiation_field_multifreq_multisource(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False, ray_batch_size=None,
):
    """
    Sum multi-frequency radiation fields from multiple point sources.

    Parameters
    ----------
    j_map            : (Nx, Ny, Nz, n_freq)
    kappa_map        : (Nx, Ny, Nz, n_freq)
    source_positions : (N_sources, 3) or list of 3-tuples, grid coordinates

    Returns
    -------
    J : (Nx, Ny, Nz, n_freq)
    """
    J_total = jnp.zeros_like(j_map)
    for pos in source_positions:
        J_total = J_total + compute_radiation_field_multifreq(
            j_map, kappa_map,
            source_pos=jnp.array(pos),
            num_rays=num_rays,
            step_size=step_size,
            max_steps=max_steps,
            use_sharding=use_sharding,
            ray_batch_size=ray_batch_size,
        )
    return J_total
