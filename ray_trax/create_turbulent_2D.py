import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

def generate_correlated_lognormal_field(
    key,
    shape=(100, 100),
    mean=1.0,
    length_scale=0.1,
    sigma_g=1.0,
    percentile=99
):
    """
    Generate a 2D log-normal random field with spatial correlations.

    Args:
        key: JAX PRNG key.
        shape: tuple (Nx, Ny) of field dimensions.
        mean: desired mean of the real-space log-normal field.
        length_scale: controls correlation (smaller = more small-scale structure).
        sigma_g: std dev of the Gaussian log field (controls contrast).
        percentile: used to return a mask of "top X%" regions.

    Returns:
        field: 2D log-normal field with spatial correlation and given mean.
        top_mask: binary mask of top percentile pixels (e.g., top 1%)
    """
    Nx, Ny = shape
    key, subkey = random.split(key)

    # --- Step 1: define k-space grid
    kx = jnp.fft.fftfreq(Nx) / length_scale
    ky = jnp.fft.fftfreq(Ny) / length_scale
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')
    k = jnp.sqrt(kx_grid**2 + ky_grid**2)

    # --- Step 2: Power spectrum (Gaussian in log-k)
    k0 = 1.0
    log_k = jnp.log(jnp.clip(k, a_min=1e-6))  # avoid log(0)
    log_k0 = jnp.log(k0)
    sigma_k = 0.5
    P_k = jnp.exp(-0.5 * ((log_k - log_k0) / sigma_k)**2)
    P_k = P_k.at[0, 0].set(0.0)  # zero DC

    # --- Step 3: Generate Gaussian field in Fourier space
    phases = jnp.exp(2j * jnp.pi * random.uniform(subkey, (Nx, Ny)))
    amplitude = jnp.sqrt(P_k)
    fft_field = amplitude * phases

    # Hermitian symmetry for real field
    if Nx % 2 == 0:
        fft_field = fft_field.at[Nx // 2, :].set(fft_field[Nx // 2, :].real)
    if Ny % 2 == 0:
        fft_field = fft_field.at[:, Ny // 2].set(fft_field[:, Ny // 2].real)
    ix = jnp.arange(0, Nx // 2)
    iy = jnp.arange(0, Ny // 2)
    fft_field = fft_field.at[-ix[:, None], -iy[None, :]].set(jnp.conj(fft_field[ix[:, None], iy[None, :]]))

    # --- Step 4: Inverse FFT → correlated Gaussian field
    g = jnp.fft.ifft2(fft_field).real
    g = (g - jnp.mean(g)) / jnp.std(g)  # normalize to mean=0, std=1
    g = sigma_g * g

    # --- Step 5: Exponentiate to log-normal
    lognormal_field = jnp.exp(g)

    # --- Step 6: Rescale to desired mean
    current_mean = jnp.mean(lognormal_field)
    field = lognormal_field * (mean / current_mean)

    # --- Step 7: Create top-X% mask
    threshold = jnp.percentile(field, percentile)
    top_mask = field >= threshold

    return field, top_mask



key = random.PRNGKey(0)
field, mask = generate_correlated_lognormal_field(
    key,
    shape=(200, 200),
    mean=1.0,
    length_scale=0.1,
    sigma_g=1.5,
    percentile=99
)

# Plot field and top 1% mask
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(jnp.log10(field), origin='lower', cmap='viridis')
plt.title("Correlated Log-Normal Field")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(mask, origin='lower', cmap='Reds')
plt.title("Top 1% Fluctuation Mask")
plt.colorbar()

plt.tight_layout()
plt.show()

