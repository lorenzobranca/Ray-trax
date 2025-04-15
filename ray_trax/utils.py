import jax
import jax.numpy as jnp


import jax.numpy as jnp

def gaussian_emissivity(Nx, Ny, Nz=None, center=(0, 0, 0), amplitude=1e3, width=5.0):
    """
    Generate a 2D or 3D Gaussian emissivity field centered at a given point.

    Parameters:
        Nx, Ny, Nz (int): Grid sizes. Nz can be None for 2D.
        center (tuple): Coordinates of the center (2D or 3D).
        amplitude (float): Peak emissivity.
        width (float): Gaussian width (standard deviation).

    Returns:
        jnp.ndarray: Emissivity map (2D or 3D)
    """
    if Nz is None:
        # 2D version
        X, Y = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), indexing='ij')
        x0, y0 = center
        r2 = (X - x0)**2 + (Y - y0)**2
        return amplitude * jnp.exp(-r2 / (2 * width**2))
    else:
        # 3D version
        X, Y, Z = jnp.meshgrid(
            jnp.arange(Nx), jnp.arange(Ny), jnp.arange(Nz),
            indexing='ij'
        )
        x0, y0, z0 = center
        r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
        return amplitude * jnp.exp(-r2 / (2 * width**2))

