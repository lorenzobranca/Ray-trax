import jax
import jax.numpy as jnp
from functools import partial

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
    x0 = jnp.floor(x).astype(int)
    y0 = jnp.floor(y).astype(int)
    z0 = jnp.floor(z).astype(int)
    x1 = jnp.clip(x0 + 1, 0, Nx - 1)
    y1 = jnp.clip(y0 + 1, 0, Ny - 1)
    z1 = jnp.clip(z0 + 1, 0, Nz - 1)
    x0 = jnp.clip(x0, 0, Nx - 1)
    y0 = jnp.clip(y0, 0, Nz - 1)
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
@partial(jax.jit, static_argnames=[
    "num_rays", "step_size", "use_sharding", "max_steps", "radiation_velocity", "step_size"
])

def compute_radiation_field_from_source_with_time_step(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5,
    max_steps = None,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False
):
    """
    Compute the radiation field within one time step (propagation distance = c * dt).
    """
    Nx, Ny, Nz = j_map.shape
    directions = sample_sphere(num_rays)
    
    if max_steps is None:
        max_distance = radiation_velocity * time_step
        max_steps = jnp.ceil(max_distance / step_size).astype(int)


    def trace_single_ray(direction):
        x, y, z = source_pos
        I = 0.0
        tau = 0.0
        J = jnp.zeros_like(j_map)

        def body_fn(i, state):
            x, y, z, I, tau, J = state
            j_val = trilinear_op(j_map, x, y, z, mode="interp")
            kappa_val = trilinear_op(kappa_map, x, y, z, mode="interp") #maybe issue, see possible fixing below
            #ix = jnp.clip(jnp.floor(x).astype(int), 0, Nx - 1)
            #iy = jnp.clip(jnp.floor(y).astype(int), 0, Ny - 1)
            #iz = jnp.clip(jnp.floor(y).astype(int), 0, Nz - 1)

            #kappa_val = kappa_map[ix,iy,iz]

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
        return final_state[-1]  # Return J

    if use_sharding:
        # Placeholder: implement sharding if needed
        raise NotImplementedError("Sharding not implemented yet.")
    else:
        J_all = jax.vmap(trace_single_ray)(directions)

    return jnp.sum(J_all, axis=0) * (4 * jnp.pi / num_rays)



def compute_radiation_field_from_multiple_sources_with_time_step(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False
):
    """
    Compute the radiation field within one time step from multiple sources.
    """
    I_total = jnp.zeros_like(j_map)
    for source_pos in source_positions:
        I_total += compute_radiation_field_from_source_with_time_step(
            j_map, kappa_map, source_pos,
            num_rays=num_rays,
            step_size=step_size,
            radiation_velocity=radiation_velocity,
            time_step=time_step,
            use_sharding=use_sharding
        )
    return I_total

