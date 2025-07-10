import jax
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

def process_density_field(kappa, percentile=99.99, amplitude=1e3, width=1.0):
    """
    Processes a 3D density field to extract the top percentile regions
    and creates an emissivity field by injecting Gaussian sources.

    Parameters:
    - kappa (jnp.ndarray): 3D density field
    - percentile (float): threshold percentile to define star regions
    - amplitude (float): peak amplitude of Gaussian emissivity
    - width (float): width of the Gaussian

    Returns:
    - emissivity (jnp.ndarray): 3D emissivity field
    - mask (jnp.ndarray): boolean mask of star regions
    - star_positions (jnp.ndarray): float32 coordinates of star centers
    """
    Nx, Ny, Nz = kappa.shape

    # Compute the mask for high-density regions
    threshold = jnp.percentile(kappa, percentile)
    mask = kappa >= threshold

    # Get star positions
    star_indices = jnp.argwhere(mask)
    star_positions = star_indices.astype(jnp.float32) + 0.5  # center in the cell

    # Initialize emissivity field
    emissivity = jnp.zeros_like(kappa)

    # Inject Gaussian sources
    for pos in star_positions:
        emissivity += gaussian_emissivity(Nx, Ny, Nz, center=pos, amplitude=amplitude, width=width)

    return emissivity, mask, star_positions
