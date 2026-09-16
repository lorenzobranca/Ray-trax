"""
Time-dependent 3D ray tracer: one horizon update of length c·Δt from point sources.

Rays are marched for N_s = ceil(c·Δt / Δs) steps (or an explicit `max_steps`).
Thin wrapper around the shared primitives in `ray_trax.core`.
"""

from functools import partial

import jax
import jax.numpy as jnp

from ray_trax.core import sample_sphere, trace_ray, sum_over_rays


@partial(
    jax.jit,
    static_argnames=[
        "num_rays", "step_size", "use_sharding", "max_steps",
        "radiation_velocity", "ray_batch_size",
    ],
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
    One time-horizon update from a single source.

    Parameters
    ----------
    j_map, kappa_map   : (Nx, Ny, Nz)
    source_pos         : (3,) grid coordinates
    num_rays           : int   Fibonacci-lattice directions
    step_size          : float ray step in grid units
    max_steps          : int | None  steps per ray; None -> ceil(c·Δt / Δs).
                         Pass an int for a static loop count (needed for reverse-mode autodiff).
    radiation_velocity : c in grid units per time unit (static)
    time_step          : Δt; horizon is c·Δt
    use_sharding       : shard rays across devices; num_rays must be divisible by n_devices
    ray_batch_size     : int | None  micro-batch rays to cap peak memory

    Returns
    -------
    J : (Nx, Ny, Nz)
    """
    directions = sample_sphere(num_rays)

    if max_steps is None:
        max_distance = jnp.asarray(radiation_velocity) * jnp.asarray(time_step)
        max_steps = jnp.ceil(max_distance / jnp.asarray(step_size)).astype(jnp.int32)

    trace = partial(trace_ray, j_map, kappa_map, source_pos,
                    step_size=step_size, max_steps=max_steps)
    J_sum = sum_over_rays(trace, directions, num_rays=num_rays,
                          use_sharding=use_sharding, ray_batch_size=ray_batch_size)
    return J_sum * (4.0 * jnp.pi / num_rays)


def compute_radiation_field_from_multiple_sources_with_time_step(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False, unroll=False,
    ray_batch_size=None,
):
    """
    One time-horizon update from multiple sources, accumulated with lax.fori_loop.

    source_positions : (N_sources, 3) array of grid coordinates
    unroll           : forwarded to lax.fori_loop
    other arguments  : as in `compute_radiation_field_from_source_with_time_step`
    """

    def body_fn(i, I_total):
        I_new = compute_radiation_field_from_source_with_time_step(
            j_map, kappa_map, source_positions[i],
            num_rays=num_rays,
            step_size=step_size,
            radiation_velocity=radiation_velocity,
            time_step=time_step,
            use_sharding=use_sharding,
            ray_batch_size=ray_batch_size,
        )
        return I_total + I_new

    num_sources = source_positions.shape[0]
    I_total = jnp.zeros_like(j_map)
    return jax.lax.fori_loop(0, num_sources, body_fn, I_total, unroll=unroll)
