import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt



from ray_trax.create_turbulent_2D import generate_correlated_lognormal_field
from ray_trax.ray_trax_2D import (
    compute_radiation_field_from_multiple_sources
)
from ray_trax.utils import gaussian_emissivity
import time


# Config
Nx, Ny = 200, 200
key = random.PRNGKey(111)

# Generate absorption field and mask
kappa, mask = generate_correlated_lognormal_field(
    key, shape=(Nx, Ny),
    mean=1.0, length_scale=0.05,
    sigma_g=0.8, percentile=99
)

##### remove it!!!

#kappa0 = 0.01
#kappa = jnp.ones((Nx,Ny)) * kappa0

#####

# Get star positions
star_indices = jnp.argwhere(mask)
star_positions = star_indices.astype(jnp.float32) + 0.5

# Create emissivity field
emissivity = jnp.zeros((Nx, Ny))
for pos in star_positions:
    emissivity += gaussian_emissivity(Nx, Ny, center=pos, amplitude=1e3, width=0.5)

# Ray tracing
tstart = time.time()
J_multi = compute_radiation_field_from_multiple_sources(
    emissivity, kappa, star_positions,
    num_rays=720, step_size=0.2, max_steps=1500
)
tend = time.time()
print("Total time: ", tend - tstart)

# Plot
plt.figure(figsize=(6, 5))
plt.imshow(jnp.log10(J_multi + 1e-6), origin='lower', cmap='inferno', vmax = 6.5)
plt.title("Ray-Traced Intensity Image")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="log10(Intensity)")
plt.tight_layout()
plt.savefig('plots/output.png')
plt.show()

