import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["num_rays", "step_size", "max_steps"])
def compute_radiation_field_from_source(j_map, kappa_map, source_pos, num_rays=1000, step_size=0.5, max_steps=500):

    """
    Compute the radiation field in 3D space from a single distributed source via ray tracing.

    Parameters:
        j_map (3D array): emissivity field
        kappa_map (3D array): opacity field
        source_pos (array-like of shape (3,)): starting point of rays
        num_rays (int): number of directions to sample
        step_size (float): step size along rays
        max_steps (int): max steps per ray

    Returns:
        3D array: accumulated radiation field
    """

    Nx, Ny, Nz = j_map.shape

    # === Sample directions uniformly on the unit sphere using the Fibonacci method ===
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

    directions = sample_sphere(num_rays)

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

    # === Trace one ray from the source in a given direction ===
    def trace_single_ray(direction):
        def body_fn(i, state):
            x, y, z, I, tau, J = state

            # Interpolate local emissivity and opacity
            j_val = trilinear_op(j_map, x, y, z, mode="interp")
            kappa_val = trilinear_op(kappa_map, x, y, z, mode="interp")

            # Step updates
            ds = step_size
            d_tau = kappa_val * ds
            dI = j_val * jnp.exp(-tau) * ds

            
            # Update intensity and optical depth
            I_new = I * jnp.exp(-d_tau) + dI
            tau_new = tau + d_tau

            # Deposit the current intensity in the radiation field
            J = trilinear_op(J, x, y, z, value=I_new, mode="deposit")

            # Move to next position along ray
            x_new = x + direction[0] * ds
            y_new = y + direction[1] * ds
            z_new = z + direction[2] * ds
            return (x_new, y_new, z_new, I_new, tau_new, J)

        # Initialize state at source
        x0, y0, z0 = source_pos
        initial = (x0, y0, z0, 0.0, 0.0, jnp.zeros_like(j_map))

        # March forward along the ray
        _, _, _, _, _, J_ray = jax.lax.fori_loop(0, max_steps, body_fn, initial)
        return J_ray

    # === Trace all rays in parallel ===
    J_all = jax.vmap(trace_single_ray)(directions)

    # Normalize by the total solid angle
    return jnp.sum(J_all, axis=0) * (4 * jnp.pi / num_rays)


def compute_radiation_field_from_multiple_sources(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5, max_steps=500
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
            max_steps=max_steps
        )
    return J_total

