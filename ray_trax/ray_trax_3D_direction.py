"""
Time-dependent 3D ray tracer that also returns a per-voxel mean direction.

Alongside the scalar field J = Σ I, the tracer accumulates the vector field
D = Σ I·n̂ over rays and returns direction_map = D / (J + eps), shape
(Nx, Ny, Nz, 3). Thin wrapper around the shared primitives in `ray_trax.core`.
"""

from functools import partial

import jax
import jax.numpy as jnp

from ray_trax.core import sample_sphere, trace_ray, sum_over_rays


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
    Time-dependent tracing from a single source over a horizon c * time_step.
    Returns the total scalar intensity J and a per-voxel direction map
    (Σ I·n̂) / (Σ I + eps).

    Returns
    -------
    J_total       : (Nx, Ny, Nz)
    direction_map : (Nx, Ny, Nz, 3)
    """
    # --- directions (reuse provided or sample once; keep unit norm) ---
    if directions is None:
        dirs = sample_sphere(num_rays).astype(jnp.float32)
    else:
        dirs = jnp.asarray(directions, dtype=jnp.float32)
        if dirs.shape != (num_rays, 3):
            raise ValueError(f"directions must have shape ({num_rays}, 3), got {dirs.shape}")
    dirs = dirs / jnp.linalg.norm(dirs, axis=-1, keepdims=True)

    # --- steps along the ray: derive from physics if not provided ---
    if max_steps is None:
        max_distance = jnp.asarray(radiation_velocity) * jnp.asarray(time_step)
        max_steps = jnp.ceil(max_distance / jnp.asarray(step_size)).astype(jnp.int32)

    trace = partial(trace_ray, j_map, kappa_map, source_pos,
                    step_size=step_size, max_steps=max_steps, with_direction=True)
    J_sum, D_sum = sum_over_rays(trace, dirs, num_rays=num_rays,
                                 use_sharding=use_sharding, ray_batch_size=ray_batch_size)

    # solid-angle weight for quasi-Monte-Carlo integration over directions
    w = (4.0 * jnp.pi) / num_rays
    J_total = J_sum * w
    D_total = D_sum * w
    direction_map = D_total / (J_total[..., None] + eps)
    return J_total, direction_map
