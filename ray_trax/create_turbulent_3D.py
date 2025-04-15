import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt


def generate_correlated_lognormal_field_3D(
    key,
    shape=(64, 64, 64),
    mean=1.0,
    length_scale=0.1,
    sigma_g=1.0,
    percentile=99
):
    """
    Generate a 3D log-normal random field with spatial correlations.

    Args:
        key: JAX PRNG key.
        shape: tuple (Nx, Ny, Nz) of field dimensions.
        mean: desired mean of the real-space log-normal field.
        length_scale: controls correlation (smaller = more small-scale structure).
        sigma_g: std dev of the Gaussian log field (controls contrast).
        percentile: threshold to return a top-percentile binary mask.

    Returns:
        field: 3D log-normal field
        top_mask: binary mask of top percentile voxels
    """
    Nx, Ny, Nz = shape
    key, subkey = random.split(key)

    # Step 1: define 3D k-space grid
    kx = jnp.fft.fftfreq(Nx) / length_scale
    ky = jnp.fft.fftfreq(Ny) / length_scale
    kz = jnp.fft.fftfreq(Nz) / length_scale
    kx_grid, ky_grid, kz_grid = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k = jnp.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    # Step 2: Power spectrum (Gaussian in log-k)
    k0 = 1.0
    log_k = jnp.log(jnp.clip(k, a_min=1e-6))
    log_k0 = jnp.log(k0)
    sigma_k = 0.5
    P_k = jnp.exp(-0.5 * ((log_k - log_k0) / sigma_k)**2)
    P_k = P_k.at[0, 0, 0].set(0.0)  # zero DC component

    # Step 3: Random complex field in Fourier space
    phases = jnp.exp(2j * jnp.pi * random.uniform(subkey, (Nx, Ny, Nz)))
    amplitude = jnp.sqrt(P_k)
    fft_field = amplitude * phases

    # Step 4: IFFT to real space (Hermitian symmetry not strictly enforced in JAX)
    g = jnp.fft.ifftn(fft_field).real
    g = (g - jnp.mean(g)) / jnp.std(g)
    g = sigma_g * g

    # Step 5: Exponentiate to log-normal
    lognormal_field = jnp.exp(g)

    # Step 6: Rescale to desired mean
    current_mean = jnp.mean(lognormal_field)
    field = lognormal_field * (mean / current_mean)

    # Step 7: Create top-X% mask
    threshold = jnp.percentile(field, percentile)
    top_mask = field >= threshold

    return field, top_mask


key = random.PRNGKey(42)
field3D, mask3D = generate_correlated_lognormal_field_3D(
    key,
    shape=(64, 64, 64),
    mean=1.0,
    length_scale=0.1,
    sigma_g=1.5,
    percentile=99
)


plt.imshow(jnp.log10(field3D[:, :, 32]), origin='lower', cmap='viridis')
plt.title("Slice of 3D Log-Normal Field")
plt.colorbar()
plt.show()
