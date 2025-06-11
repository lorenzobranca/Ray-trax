import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # or whatever GPU you prefer

import time
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

from ray_trax.create_turbulent_3D import generate_correlated_lognormal_field_3D
from ray_trax.utils import gaussian_emissivity
from ray_trax.ray_trax_3D_tdep import (
    compute_radiation_field_from_multiple_sources_with_time_step
)

# Config
Nx, Ny, Nz = 64, 64, 64
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
emissivity = jnp.zeros((Nx, Ny, Nz))
for pos in star_positions:
    emissivity += gaussian_emissivity(Nx, Ny, Nz, center=pos, amplitude=1e3, width=1.0)

# Time-stepping parameters
total_time = 10.0
dt = 200.0
c = 1.0  # Speed of light in code units

# Initialize cumulative radiation field
J_total = jnp.zeros((Nx, Ny, Nz))

tstart = time.time()

'''
for step in range(int(total_time / dt)):
    print(f"Time step {step + 1}/{int(total_time / dt)}")

    # Here you could update kappa or emissivity dynamically if needed
    # e.g. kappa = update_kappa(...)
    # e.g. emissivity = update_emissivity(...)

    J_step = compute_radiation_field_from_multiple_sources_with_time_step(
        emissivity, kappa, star_positions,
        num_rays=4096,
        step_size=0.5,
        radiation_velocity=c,
        time_step=dt
    )
    J_total += J_step
'''

J_total = compute_radiation_field_from_multiple_sources_with_time_step(
        emissivity, kappa, star_positions,
        num_rays=4096,
        step_size=0.5,
        radiation_velocity=c,
        time_step=dt
    )


tend = time.time()
print("Total simulation time:", tend - tstart)


output_dir = 'plots_3d_time_dep'
os.makedirs(output_dir, exist_ok=True)

# Plot results
mid_x = Nx // 2
mid_y = Ny // 2
mid_z = Nz // 2

# Plot x-y plane at fixed z
plt.figure(figsize=(6, 5))
plt.imshow(jnp.log10(J_total[:, :, mid_z] + 1e-6), origin='lower', cmap='inferno')
plt.title(f"Cumulative X-Y plane at Z={mid_z}")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="log10(Intensity)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'xy_plane.png'))
plt.close()

# Plot x-z plane at fixed y
plt.figure(figsize=(6, 5))
plt.imshow(jnp.log10(J_total[:, mid_y, :] + 1e-6), origin='lower', cmap='inferno', aspect=Nz/Nx)
plt.title(f"Cumulative X-Z plane at Y={mid_y}")
plt.xlabel("X")
plt.ylabel("Z")
plt.colorbar(label="log10(Intensity)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'xz_plane.png'))
plt.close()

# Plot y-z plane at fixed x
plt.figure(figsize=(6, 5))
plt.imshow(jnp.log10(J_total[mid_x, :, :] + 1e-6), origin='lower', cmap='inferno', aspect=Nz/Ny)
plt.title(f"Cumulative Y-Z plane at X={mid_x}")
plt.xlabel("Y")
plt.ylabel("Z")
plt.colorbar(label="log10(Intensity)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'yz_plane.png'))
plt.close()

print("All plots saved to", output_dir)

