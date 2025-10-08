import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8" #, 9"

import sys
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt



from ray_trax.create_turbulent_3D import generate_correlated_lognormal_field_3D
from ray_trax.ray_trax_3D import compute_radiation_field_from_multiple_sources
from ray_trax.utils import gaussian_emissivity
import time


# Config
Nx, Ny, Nz = 128, 128, 128
key = random.PRNGKey(111)

# Generate absorption field and mask
kappa, mask = generate_correlated_lognormal_field_3D(
    key, shape=(Nx, Ny, Nz),
    mean=1.0, length_scale=0.05,
    sigma_g=1.2, percentile=99.5
)



# Get star positions
star_indices = jnp.argwhere(mask)
star_positions = star_indices.astype(jnp.float32) + 0.5

# Create emissivity field
emissivity = jnp.zeros((Nx, Ny))
for pos in star_positions:
    emissivity += gaussian_emissivity(Nx, Ny, Nz, center=pos, amplitude=1e3, width=1.0)

# Ray tracing
tstart = time.time()
J_multi = compute_radiation_field_from_multiple_sources(
    emissivity, kappa, star_positions,
    num_rays=4096, step_size=0.5, max_steps=400, use_sharding = True
)
tend = time.time()
print("Total time: ", tend - tstart)

output_dir = 'plots_3d'

# Select slices to plot
mid_x = Nx // 2
mid_y = Ny // 2
mid_z = Nz // 2

# Plot x-y plane at fixed z
plt.figure(figsize=(6, 5))
plt.imshow(jnp.log10(J_multi[:, :, mid_z] + 1e-6), origin='lower', cmap='inferno')
plt.title(f"X-Y plane at Z={mid_z}")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="log10(Intensity)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'xy_plane.png'))
plt.close()

# Plot x-z plane at fixed y
plt.figure(figsize=(6, 5))
plt.imshow(jnp.log10(J_multi[:, mid_y, :] + 1e-6), origin='lower', cmap='inferno', aspect=Nz/Nx)
plt.title(f"X-Z plane at Y={mid_y}")
plt.xlabel("X")
plt.ylabel("Z")
plt.colorbar(label="log10(Intensity)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'xz_plane.png'))
plt.close()

# Plot y-z plane at fixed x
plt.figure(figsize=(6, 5))
plt.imshow(jnp.log10(J_multi[mid_x, :, :] + 1e-6), origin='lower', cmap='inferno', aspect=Nz/Ny)
plt.title(f"Y-Z plane at X={mid_x}")
plt.xlabel("Y")
plt.ylabel("Z")
plt.colorbar(label="log10(Intensity)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yz_plane.png'))
plt.close()

