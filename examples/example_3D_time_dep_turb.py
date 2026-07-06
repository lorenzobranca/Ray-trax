import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import gc

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import imageio

#from ray_trax.create_turbulent_3D import generate_correlated_lognormal_field_3D
from ray_trax.utils import gaussian_emissivity, process_density_field
from ray_trax.ray_trax_3D_tdep import (
    compute_radiation_field_from_multiple_sources_with_time_step
)

# Config
Nx, Ny, Nz = 128, 128, 128
key = random.PRNGKey(111)


# Load your saved 3D field (replace with your actual load logic)
kappa = jnp.load("./turbulent_fields/0_s-1.7_rms20p0.npy")

plt.imshow(kappa[:,:,64])
plt.savefig('turb_field.png')


# Process it
emissivity, mask, star_positions = process_density_field(kappa, percentile=99.99)

# Time-stepping parameters
total_time = 60.0
dt = 3.0
c = 1.0  # Speed of light in code units

output_dir = 'plots_3d_time_dep'
os.makedirs(output_dir, exist_ok=True)

# --- configuration for the 3x2 panel (bin #2 = third bin) ---
selected_steps = [2, 4, 8, 11, 15, 19]   # 1-based steps
bin_index = 2                              # zero-based -> third bin
n_steps_total = int(total_time / dt)
snapshots_bin2 = {}                        # step -> 2D log10 slice
mid_z = Nz // 2

tstart = time.time()
filenames = []

for step in range(n_steps_total):
    print(f"Time step {step + 1}/{n_steps_total}")

    J_step = compute_radiation_field_from_multiple_sources_with_time_step(
        emissivity, kappa, star_positions,
        num_rays=int(6*512),  # 4096
        step_size=0.5,
        radiation_velocity=c,
        time_step=dt * (step + 1),
        use_sharding=True
    )

    # Force evaluation and break JAX graph
    J_step.block_until_ready()
    J_step = np.array(J_step)

    # ----- per-step snapshot (mid-Z, total field) -----
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log10(J_step[:, :, mid_z] + 1e-6), origin='lower', cmap='inferno')
    plt.title(f"X-Y plane at Z={mid_z} - Time step {step+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="log10(I)")
    plt.tight_layout()
    filename = os.path.join(output_dir, f'step_{step:03d}.png')
    plt.savefig(filename)
    filenames.append(filename)
    plt.close()

    # ----- store bin-2 mid-Z slice for the selected steps -----
    s = step + 1  # 1-based
    if s in selected_steps:
        if J_step.ndim == 4:
            # Accept (Nx,Ny,Nz,B) or (B,Nx,Ny,Nz)
            if J_step.shape[-1] > bin_index and J_step.shape[2] == Nz:
                # layout: (Nx, Ny, Nz, B)
                Jb = J_step[:, :, mid_z, bin_index]
            elif J_step.shape[0] > bin_index and J_step.shape[1] == Nx:
                # layout: (B, Nx, Ny, Nz)
                Jb = J_step[bin_index, :, :, mid_z]
            else:
                raise ValueError(
                    f"Unexpected per-bin shape {J_step.shape}; "
                    "expected (Nx,Ny,Nz,B) or (B,Nx,Ny,Nz)."
                )
            snapshots_bin2[s] = np.log10(Jb + 1e-6)
        else:
            print("[warning] J_step has no bin dimension (ndim != 4); "
                  "cannot extract a per-bin panel for this step.")

    # Free memory
    del J_step
    gc.collect()
    jax.clear_caches()

tend = time.time()
print("Total simulation time:", tend - tstart)

# ----- Create GIF from per-step PNGs -----
gif_filename = os.path.join(output_dir, 'radiation_evolution.gif')
with imageio.get_writer(gif_filename, mode='I', duration=0.5, loop=0) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print("GIF saved to", gif_filename)

# ----- Build the 3x2 panel for frequency bin #2 (third bin) -----
missing = [s for s in selected_steps if s not in snapshots_bin2]
if missing:
    print(f"[warning] missing snapshots for steps {missing}; panel will skip them.")

if snapshots_bin2:
    # Use a common color scale across subplots
    vals = [snapshots_bin2[s] for s in selected_steps if s in snapshots_bin2]
    vmin = min(float(x.min()) for x in vals)
    vmax = max(float(x.max()) for x in vals)

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
            # time label: fraction of total steps in decimal with 2 digits (e.g., 3/20 -> 0.15)
            tfrac = s / n_steps_total
            ax.text(
                0.02, 0.98, f"t = {tfrac:.2f}",
                color='white', ha='left', va='top',
                fontsize=11, transform=ax.transAxes,
                bbox=dict(facecolor='black', alpha=0.25, pad=2, edgecolor='none')
            )
            ax.set_title(f"step {s}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            ax.axis('off')

    # Single shared colorbar
    if 'im' in locals():
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.88, pad=0.02)
        cbar.set_label("log10(Intensity) in bin #2")

    panel_path = os.path.join(output_dir, "bin2_steps_3x2.png")
    plt.savefig(panel_path, dpi=180)
    plt.close(fig)
    print("Saved 3x2 bin-2 panel to", panel_path)
else:
    print("[info] No bin-2 snapshots collected; panel not created.")


