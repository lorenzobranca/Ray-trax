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
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    z0 = jnp.floor(z).astype(jnp.int32)

    x1 = jnp.clip(x0 + 1, 0, Nx - 1)
    y1 = jnp.clip(y0 + 1, 0, Ny - 1)
    z1 = jnp.clip(z0 + 1, 0, Nz - 1)
    x0 = jnp.clip(x0, 0, Nx - 1)
    y0 = jnp.clip(y0, 0, Ny - 1)  # <-- fixed
    z0 = jnp.clip(z0, 0, Nz - 1)

    dx, dy, dz = (x - x0), (y - y0), (z - z0)

    w000 = (1-dx)*(1-dy)*(1-dz); w001 = (1-dx)*(1-dy)*dz
    w010 = (1-dx)*dy*(1-dz);     w011 = (1-dx)*dy*dz
    w100 = dx*(1-dy)*(1-dz);     w101 = dx*(1-dy)*dz
    w110 = dx*dy*(1-dz);         w111 = dx*dy*dz

    if mode == "interp":
        v000 = grid[x0, y0, z0]; v001 = grid[x0, y0, z1]
        v010 = grid[x0, y1, z0]; v011 = grid[x0, y1, z1]
        v100 = grid[x1, y0, z0]; v101 = grid[x1, y0, z1]
        v110 = grid[x1, y1, z0]; v111 = grid[x1, y1, z1]
        return (w000*v000 + w001*v001 + w010*v010 + w011*v011 +
                w100*v100 + w101*v101 + w110*v110 + w111*v111)

    elif mode == "deposit":
        assert value is not None
        out = grid
        out = out.at[x0, y0, z0].add(w000 * value)
        out = out.at[x0, y0, z1].add(w001 * value)
        out = out.at[x0, y1, z0].add(w010 * value)
        out = out.at[x0, y1, z1].add(w011 * value)
        out = out.at[x1, y0, z0].add(w100 * value)
        out = out.at[x1, y0, z1].add(w101 * value)
        out = out.at[x1, y1, z0].add(w110 * value)
        out = out.at[x1, y1, z1].add(w111 * value)
        return out

    else:
        raise ValueError("mode must be 'interp' or 'deposit'")

