"""
3D multi-frequency ray-tracing solver for the RTE.

Spectral extension of ray_trax_3D.py: j_map, kappa_map and the output J carry
a trailing frequency axis of length n_freq. Inside each ray step I and tau are
(n_freq,) vectors; all operations are elementwise over frequency. Frequency
bins are independent (no scattering). The shared primitives in `ray_trax.core`
broadcast over the trailing axis, so this module is a thin wrapper.

Inputs
------
j_map     : (Nx, Ny, Nz, n_freq)   emissivity  [e.g. erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹]
kappa_map : (Nx, Ny, Nz, n_freq)   opacity     [cm⁻¹]
            built with cross_sections.build_kappa_map()

Output
------
J         : (Nx, Ny, Nz, n_freq)   mean intensity, normalised by 4π / num_rays
"""

from functools import partial

import jax
import jax.numpy as jnp

from ray_trax.core import sample_sphere, trace_ray, sum_over_rays


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
    directions = sample_sphere(num_rays)
    trace = partial(trace_ray, j_map, kappa_map, source_pos,
                    step_size=step_size, max_steps=max_steps, kappa_interp=kappa_interp)
    J_sum = sum_over_rays(trace, directions, num_rays=num_rays,
                          use_sharding=use_sharding, ray_batch_size=ray_batch_size)
    return J_sum * (4 * jnp.pi / num_rays)


def compute_radiation_field_multifreq_multisource(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False, ray_batch_size=None,
    kappa_interp="nearest",
):
    """
    Sum multi-frequency radiation fields from multiple point sources.

    Parameters
    ----------
    j_map            : (Nx, Ny, Nz, n_freq)
    kappa_map        : (Nx, Ny, Nz, n_freq)
    source_positions : (N_sources, 3) or list of 3-tuples, grid coordinates
    kappa_interp     : "nearest" or "trilinear", forwarded to compute_radiation_field_multifreq

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
            kappa_interp=kappa_interp,
        )
    return J_total
