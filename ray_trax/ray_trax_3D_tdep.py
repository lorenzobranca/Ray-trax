import jax
import jax.numpy as jnp
import numpy as np
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

def trilinear_deposit(grid, x, y, z, value):
    Nx, Ny, Nz = grid.shape
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

    indices = [
        (x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1),
        (x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1)
    ]

    for w, (ix, iy, iz) in zip(weights, indices):
        grid = grid.at[ix, iy, iz].add(w * value)
    return grid

@partial(jax.jit, static_argnames=["num_rays", "step_size", "max_steps", "nt"])
def compute_time_dependent_radiation_field(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5, max_steps=500,
    nt=100  # number of time steps
):
    Nx, Ny, Nz = j_map.shape
    directions = sample_sphere(num_rays)
    I_grid = jnp.zeros((nt, Nx, Ny, Nz))

    def trace_single_ray(direction):
        x, y, z = source_pos
        I = 0.0
        tau = 0.0
        intensity_steps = []

        for t in range(nt):
            j_val = j_map[int(jnp.clip(x, 0, Nx-1)),
                          int(jnp.clip(y, 0, Ny-1)),
                          int(jnp.clip(z, 0, Nz-1))]
            kappa_val = kappa_map[int(jnp.clip(x, 0, Nx-1)),
                                  int(jnp.clip(y, 0, Ny-1)),
                                  int(jnp.clip(z, 0, Nz-1))]
            ds = step_size
            d_tau = kappa_val * ds
            dI = j_val * jnp.exp(-tau) * ds
            I = I * jnp.exp(-d_tau) + dI
            tau += d_tau
            intensity_steps.append((t, x, y, z, I))

            # Move ray forward
            x += direction[0] * ds
            y += direction[1] * ds
            z += direction[2] * ds

        return intensity_steps

    # Run for all rays
    all_ray_data = jax.vmap(trace_single_ray)(directions)

    # Deposit into time-dependent grid
    for ray_steps in all_ray_data:
        for t, x, y, z, Ival in ray_steps:
            I_grid = trilinear_deposit(I_grid.at[t], x, y, z, Ival)

    return I_grid


def compute_radiation_field_from_multiple_sources(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5, max_steps=500,
    use_sharding=False
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
        use_sharding (bool): whether to use multi-GPU parallelization

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
                max_steps=max_steps,
                use_sharding = use_sharding

            )

    return J_total

