# main_3D_time_dep_turb_multinu.py
import os
from autocvd import autocvd
autocvd(num_gpus=1)   # pick a free GPU; must run before importing jax
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
num_rays = 3500
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

# --- NEW: configuration for the 3×2 panel of bin #2 (the third) ---
selected_steps = [5, 10, 25, 32, 45, 55]   # 1-based indices
bin_of_interest = 2                        # zero-based -> 3rd bin
n_steps_total = int(total_time / dt)
mid_z = Nz // 2
snapshots_bin2 = {}                        # step -> 2D log10 slice

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

        # --- NEW: capture mid-Z slice for the 3×2 panel (only bin #2 at specific steps) ---
        s = step + 1  # 1-based for readability
        if (b == bin_of_interest) and (s in selected_steps):
            snapshots_bin2[s] = np.log10(J_b_np[:, :, mid_z] + 1e-6)

        # cleanup per bin
        del J_b, J_b_np
        gc.collect()
        jax.clear_caches()

    # ---- Plot combined snapshot ----
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(J_total[:, :, mid_z] + 1e-6), origin='lower', cmap='inferno')
    plt.title(f"Total Intensity ({num_bins} bins) | Z={mid_z} | step {step+1}")
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
        os.makedirs(os.path.join(output_dir, "per_bin"), exist_ok=True)
        for b, Jb in enumerate(J_bins_np):
            fig, ax = plt.subplots(figsize=(6.5, 6))
            im = ax.imshow(np.log10(Jb[:, :, mid_z] + 1e-6),
                           origin='lower', cmap='inferno')

            # Thicker frame (spines)
            for spine in ax.spines.values():
                spine.set_linewidth(2.2)

            # Thicker ticks, but no tick labels
            ax.tick_params(axis='both', length=8, width=2.2,
                           labelbottom=False, labelleft=False)

            # No axis labels
            ax.set_xlabel(None)
            ax.set_ylabel(None)

            # Colorbar with thicker ticks, but no label
            cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95, label="log10(I)")
            cbar.ax.tick_params(length=6, width=2.0)

            plt.tight_layout()
            fname_b = os.path.join(output_dir, "per_bin", f"step_{step:03d}_bin_{b+1}.png")
            plt.savefig(fname_b, dpi=200, bbox_inches="tight")
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

# ------------------------------------------------------
# NEW: Build the 3×2 panel for frequency bin #2 (third)
# ------------------------------------------------------
missing = [s for s in selected_steps if s not in snapshots_bin2]
if missing:
    print(f"[warning] Missing snapshots for steps {missing}; the panel will skip them.")

if snapshots_bin2:
    # Use a common color scale for comparability
    vals = [snapshots_bin2[s] for s in selected_steps if s in snapshots_bin2]
    vmin = min(float(x.min()) for x in vals)
    vmax = max(float(x.max()) for x in vals)

    # 2 rows × 3 columns (i.e., 3×2 grid of six subplots)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    axes = axes.ravel()

    for i, s in enumerate(selected_steps):
        ax = axes[i]
        if s in snapshots_bin2:
            im = ax.imshow(
                snapshots_bin2[s],
                origin='lower',
                cmap='inferno',
                vmin=vmin,
                vmax=vmax
            )
            # time label: fraction of total steps, decimal with **two significant digits**
            tfrac = s / n_steps_total
            tfrac_str = f"{tfrac:.2g}"   # e.g., 3/20 -> 0.15 (two sig. digits)
            ax.text(
                0.02, 0.98, f"t = {tfrac_str}",
                color='white', ha='left', va='top',
                fontsize=11, transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.25, pad=2, edgecolor='none')
            )
            #ax.set_title(f"step {s}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            ax.axis('off')

    if 'im' in locals():
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.88, pad=0.02)
        cbar.set_label("log10(Intensity)  (bin #2)")

    panel_path = os.path.join(output_dir, "bin2_steps_3x2.png")
    plt.savefig(panel_path, dpi=180)
    plt.close(fig)
    print(f"Saved 3×2 bin-2 panel to {panel_path}")
else:
    print("[info] No bin-2 snapshots collected; 3×2 panel not created.")


