"""
Steady-state 3D ray tracer: mean intensity from point sources on a regular grid.

Thin wrapper around the shared primitives in `ray_trax.core`; see `trace_ray`
there for the per-step update.
"""

from functools import partial

import jax
import jax.numpy as jnp

from ray_trax.core import sample_sphere, trace_ray, sum_over_rays


@partial(jax.jit, static_argnames=["num_rays", "step_size", "max_steps", "use_sharding", "ray_batch_size"])
def compute_radiation_field_from_source(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False, ray_batch_size=None,
):
    """
    Mean radiation field J from a single point source.

    Parameters
    ----------
    j_map, kappa_map : (Nx, Ny, Nz)  emissivity and opacity
    source_pos       : (3,)  source position in grid coordinates
    num_rays         : int   Fibonacci-lattice directions
    step_size        : float ray step in grid units
    max_steps        : int   steps per ray
    use_sharding     : bool  shard rays across devices; num_rays must be divisible by n_devices
    ray_batch_size   : int | None  micro-batch rays to cap peak memory

    Returns
    -------
    J : (Nx, Ny, Nz)
    """
    directions = sample_sphere(num_rays)
    trace = partial(trace_ray, j_map, kappa_map, source_pos,
                    step_size=step_size, max_steps=max_steps)
    J_sum = sum_over_rays(trace, directions, num_rays=num_rays,
                          use_sharding=use_sharding, ray_batch_size=ray_batch_size)
    return J_sum * (4 * jnp.pi / num_rays)


def compute_radiation_field_from_multiple_sources(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False, ray_batch_size=None,
):
    """
    Total radiation field from multiple point sources (Python loop over sources).

    Parameters
    ----------
    source_positions : (N_sources, 3) or list of 3-tuples, grid coordinates
    other arguments  : as in `compute_radiation_field_from_source`

    Returns
    -------
    J : (Nx, Ny, Nz)
    """
    J_total = jnp.zeros_like(j_map)
    for source_pos in source_positions:
        J_total += compute_radiation_field_from_source(
            j_map, kappa_map,
            source_pos=jnp.array(source_pos),
            num_rays=num_rays,
            step_size=step_size,
            max_steps=max_steps,
            use_sharding=use_sharding,
            ray_batch_size=ray_batch_size,
        )
    return J_total
