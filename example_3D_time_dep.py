import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3, 4, 5, 6, 7, 8, 9"
import gc

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import imageio

from ray_trax.create_turbulent_3D import generate_correlated_lognormal_field_3D
from ray_trax.utils import gaussian_emissivity
from ray_trax.ray_trax_3D_tdep import (
    compute_radiation_field_from_multiple_sources_with_time_step
)

# Config
Nx, Ny, Nz = 128, 128, 128
key = random.PRNGKey(111)

# Generate absorption field and mask
kappa, mask = generate_correlated_lognormal_field_3D(
    key, shape=(Nx, Ny, Nz),
    mean=1.0, length_scale=0.05,
    sigma_g=1.2, percentile=99.99
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
dt = 1.0
c = 1.0  # Speed of light in code units

output_dir = 'plots_3d_time_dep'
os.makedirs(output_dir, exist_ok=True)

tstart = time.time()
filenames = []

for step in range(int(total_time / dt)):
    print(f"Time step {step + 1}/{int(total_time / dt)}")


    J_step = compute_radiation_field_from_multiple_sources_with_time_step(
        emissivity, kappa, star_positions,
        num_rays=4096,
        step_size=0.5,
        radiation_velocity=c,
        time_step=dt * step,
        use_sharding=True
    )

    # Force evaluation and break JAX graph
    J_step.block_until_ready()
    J_step = np.array(J_step)

    # Plot snapshot
    mid_z = Nz // 2
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(J_step[:, :, mid_z] + 1e-6), origin='lower', cmap='inferno')
    plt.title(f"X-Y plane at Z={mid_z} - Time step {step+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="log10(Intensity)")
    plt.tight_layout()
    filename = os.path.join(output_dir, f'step_{step:03d}.png')
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()
    
    del J_step               # drop Python reference
    gc.collect()             # free DeviceArray objects
    jax.clear_caches()     # tear down compiled executables

tend = time.time()
print("Total simulation time:", tend - tstart)

# Create GIF
gif_filename = os.path.join(output_dir, 'radiation_evolution.gif')
with imageio.get_writer(gif_filename, mode='I', duration=0.5, loop = 0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF saved to", gif_filename)

