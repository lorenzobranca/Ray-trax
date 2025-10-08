# main_3D_time_dep_turb_multinu.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import gc
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import imageio

from ray_trax.utils import process_density_field
from ray_trax.ray_trax_3D_tdep import (
    compute_radiation_field_from_multiple_sources_with_time_step
)

# ----------------------------
# Config (edit these as needed)
# ----------------------------
Nx, Ny, Nz = 128, 128, 128
key = random.PRNGKey(112)

# Frequency-bin config
num_bins = 6
nu_min, nu_max = 1e-2, 1e+2   # arbitrary code units (relative only)
T_bb = 1.0                    # "temperature" in code units (only relative matters)
beta = .5                    # opacity ~ nu^(-beta); beta>0 => decreasing with frequency
nu_ref = None                 # None => use lowest-bin center as reference

# Ray-tracing / time stepping
total_time = 60.0
dt = 1.0
c = 1.0                       # speed of light in code units
num_rays = 4096
step_size = 0.5
use_sharding = True

# Output
output_dir = 'plots_3d_time_dep_multinu'
save_per_bin = False          # set True to also save per-bin snapshots
os.makedirs(output_dir, exist_ok=True)
if save_per_bin:
    os.makedirs(os.path.join(output_dir, "per_bin"), exist_ok=True)

# --------------------------------
# Load and preprocess the 3D field
# --------------------------------
# Load your saved 3D field (replace with your actual load logic)
kappa_base = jnp.load("./turbulent_fields/0_s-1.7_rms20p0.npy")

# Build emissivity and source positions from density (your helper)
emissivity_base, mask, star_positions = process_density_field(kappa_base, percentile=99.99, amplitude = 1e2)

# ---------------------------
# Build 6-bin BlackBody shape
# ---------------------------
def planck_nu(nu, T):
    # Dimensionless Planck-like: B_nu ~ nu^3 / (exp(nu/T) - 1)
    x = nu / T
    # clip to keep things sane (avoid overflow/underflow)
    x = np.clip(x, 1e-8, 1e8)
    return (nu ** 3) / (np.expm1(x))

# Log-spaced bin edges + centers
nu_edges = np.logspace(np.log10(nu_min), np.log10(nu_max), num_bins + 1)
nu_centers = np.sqrt(nu_edges[:-1] * nu_edges[1:])
if nu_ref is None:
    nu_ref = nu_centers[0]

# Bin-integrated weights (trapz per bin on a fine sub-grid), then normalize to sum=1
weights = []
for i in range(num_bins):
    a, b = nu_edges[i], nu_edges[i+1]
    grid = np.logspace(np.log10(a), np.log10(b), 256)
    vals = planck_nu(grid, T_bb)
    integ = np.trapz(vals, grid)
    weights.append(integ)
weights = np.array(weights)
weights = weights / weights.sum()

print("Frequency bin centers:", nu_centers)
print("Normalized BB weights:", weights)
print("Opacity scaling ~ (nu/nu_ref)^(-beta) with beta =", beta, "and nu_ref =", nu_ref)

# ---------------------------------------
# Time loop: trace each bin, sum the field
# ---------------------------------------
tstart = time.time()
filenames = []

save_per_bin = True

for step in range(int(total_time / dt)):
    print(f"Time step {step + 1}/{int(total_time / dt)}")

    # Accumulator for total intensity at this step
    J_total = None
    J_bins_np = []

    for b in range(num_bins):
        # Per-bin emissivity and absorption
        emissivity_b = emissivity_base * weights[b]
        kappa_b = kappa_base * ((nu_centers[b] / nu_ref) ** (-beta))
        print(((nu_centers[b] / nu_ref) ** (-beta)))

        # Compute radiation field for this bin at this time
        J_b = compute_radiation_field_from_multiple_sources_with_time_step(
            emissivity_b, kappa_b, star_positions,
            num_rays=num_rays,
            step_size=step_size,
            radiation_velocity=c,
            time_step=dt * step,
            use_sharding=use_sharding
        )

        # Force eval and detach from JAX for accumulation/plotting
        J_b.block_until_ready()
        J_b_np = np.array(J_b)

        if J_total is None:
            J_total = J_b_np.copy()
            
        else:
            J_total += J_b_np

        if save_per_bin:
            J_bins_np.append(J_b_np)

        # cleanup per bin
        del J_b, J_b_np
        gc.collect()
        jax.clear_caches()
    
    # ---- Plot combined snapshot ----
    mid_z = Nz // 2
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(J_total[:, :, mid_z] + 1e-6), origin='lower', cmap='inferno')
    plt.title(f"Total Intensity (6 bins) | Z={mid_z} | step {step+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="log10(Intensity)")
    plt.tight_layout()
    filename = os.path.join(output_dir, f'step_{step:03d}.png')
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

    # ---- Optionally: per-bin snapshots ----
    if save_per_bin:
        for b, Jb in enumerate(J_bins_np):
            plt.figure(figsize=(6, 5))
            plt.imshow(np.log10(Jb[:, :, mid_z] + 1e-6), origin='lower', cmap='inferno')
            plt.title(f"Bin {b+1}/{num_bins} (nu={nu_centers[b]:.3g}) | Z={mid_z} | step {step+1}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.colorbar(label="log10(Intensity)")
            plt.tight_layout()
            fname_b = os.path.join(output_dir, "per_bin", f'step_{step:03d}_bin_{b+1}.png')
            plt.savefig(fname_b)
            plt.close()

    # cleanup step accumulation
    del J_total, J_bins_np
    gc.collect()
    jax.clear_caches()

tend = time.time()
print("Total simulation time:", tend - tstart)

# -------------
# Build a GIF
# -------------
gif_filename = os.path.join(output_dir, 'radiation_evolution_multinu.gif')
with imageio.get_writer(gif_filename, mode='I', duration=0.5, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF saved to", gif_filename)

