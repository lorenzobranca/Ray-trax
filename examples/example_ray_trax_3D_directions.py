# test_analytic_multi_direction.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # change as needed

import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ray_trax.ray_trax_3D_direction import (
    compute_radiation_field_from_source_with_time_step_direction
)
from ray_trax.utils import gaussian_emissivity

# ----------------------- domain & parameters -----------------------
Nx, Ny, Nz = 64, 64, 64
kappa0 = 0.1
L_total = 1.0
num_sources = 10
width = 1.5
num_rays = 8192
step_size = 0.5
c = 1.0
dt = 50.0
eps = 1e-10

outdir = "plots_3d_multi_dir"
os.makedirs(outdir, exist_ok=True)

# Absorption field (uniform)
kappa = jnp.ones((Nx, Ny, Nz), dtype=jnp.float32) * kappa0

# ----------------------- source placement -------------------------
cx, cy, cz = Nx / 2, Ny / 2, Nz / 2
offsets = jnp.array([
    [-8, -8, -8], [ 8, -8, -8], [-8,  8, -8], [ 8,  8, -8],
    [-8, -8,  8], [ 8, -8,  8], [-8,  8,  8], [ 8,  8,  8],
    [ 0,  0,  0], [12,  0,  0],
], dtype=jnp.float32)
source_positions = jnp.stack([jnp.array([cx, cy, cz], dtype=jnp.float32) + off
                              for off in offsets], axis=0)
assert source_positions.shape == (num_sources, 3)
L_i = jnp.ones((num_sources,), dtype=jnp.float32) * (L_total / num_sources)

print("source positions:\n", source_positions)

# ----------------------- emissivity field -------------------------
# Build a single emissivity map with 10 normalized Gaussian blobs
emissivity = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32)
for i in range(num_sources):
    raw = gaussian_emissivity(Nx, Ny, Nz, center=source_positions[i],
                              amplitude=1., width=width)
    raw_sum = jnp.sum(raw)
    blob = raw * (L_i[i] / (raw_sum + 1e-20))
    emissivity = emissivity + blob

# ----------------------- ray tracing (sum of sources) -------------
# IMPORTANT: To combine per-source direction maps correctly:
#   D_total = sum_i (dir_i * J_i),   J_total = sum_i J_i,
#   dir_total = D_total / (J_total + eps).
start = time.time()
J_total = jnp.zeros_like(emissivity)
D_total = jnp.zeros((Nx, Ny, Nz, 3), dtype=emissivity.dtype)

for i in range(num_sources):
    J_i, dir_i = compute_radiation_field_from_source_with_time_step_direction(
        emissivity,
        kappa,
        source_pos=source_positions[i],
        num_rays=int(num_rays),
        step_size=float(step_size),
        radiation_velocity=float(c),
        time_step=float(dt),
        use_sharding=False,       # set True if you've configured sharding
        ray_batch_size=None       # e.g., 512/1024 to cap memory
    )
    J_total = J_total + J_i
    D_total = D_total + dir_i * J_i[..., None]

elapsed = time.time() - start
print(f"Direction tracing for {num_sources} sources done in {elapsed:.2f} s")

dir_total = D_total / (J_total[..., None] + eps)

# ----------------------- analytic comparison ----------------------
# Intensity: J_an = sum_i [ L_i / (4π r_i^2) * exp(-kappa0 * r_i) ]
# Direction: D_an = sum_i [ J_i_point * (r_vec_i / r_i) ],  dir_an = D_an / J_an
X, Y, Z = jnp.meshgrid(
    jnp.arange(Nx, dtype=jnp.float32),
    jnp.arange(Ny, dtype=jnp.float32),
    jnp.arange(Nz, dtype=jnp.float32),
    indexing='ij'
)

def point_terms(center, Lsrc):
    dx = X - center[0]; dy = Y - center[1]; dz = Z - center[2]
    r = jnp.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6
    w = (Lsrc / (4.0 * jnp.pi * r*r)) * jnp.exp(-kappa0 * r)  # scalar weight
    ux, uy, uz = dx / r, dy / r, dz / r                       # unit vector
    return w, jnp.stack([w * ux, w * uy, w * uz], axis=-1)    # (scalar, vec3)

J_an = jnp.zeros_like(J_total)
D_an = jnp.zeros_like(D_total)
for i in range(num_sources):
    w_i, v_i = point_terms(source_positions[i], L_i[i])
    J_an = J_an + w_i
    D_an = D_an + v_i

dir_an = D_an / (J_an[..., None] + eps)

# Cosine similarity between numeric and analytic directions
dot = jnp.sum(dir_total * dir_an, axis=-1)
dot = jnp.clip(dot, -1.0, 1.0)
cos_sim = dot
angle_err_deg = jnp.degrees(jnp.arccos(cos_sim + 0.0))

# ----------------------- quick scalar checks ----------------------
ix, iy, iz = int(Nx/2), int(Ny/2), int(Nz/2)
print("Center intensity (analytic) =", float(J_an[ix, iy, iz]))
print("Center intensity (numeric)  =", float(J_total[ix, iy, iz]))
print("Mean rel. error (J):", float(jnp.mean(jnp.abs((J_total - J_an) / (J_an + eps)))))
print("Mean cosine similarity (dir):", float(jnp.mean(cos_sim)))
print("Median angle error (deg):", float(jnp.nanmedian(angle_err_deg)))

# ----------------------- plotting: central slices -----------------
def save_slices():
    midx, midy, midz = Nx//2, Ny//2, Nz//2

    def slice_and_plot(arr_num, arr_an, plane, idx, name, is_log=True, cmap='inferno'):
        if plane == 'x':
            num = arr_num[idx, 1:-1, 1:-1]
            ana = arr_an[ idx, 1:-1, 1:-1]
        elif plane == 'y':
            num = arr_num[1:-1, idx, 1:-1]
            ana = arr_an[ 1:-1, idx, 1:-1]
        else:
            num = arr_num[1:-1, 1:-1, idx]
            ana = arr_an[ 1:-1, 1:-1, idx]

        rel = jnp.abs((num - ana) / (ana + eps))

        plt.figure(figsize=(12, 4))
        if is_log:
            plt.subplot(1, 3, 1); plt.imshow(jnp.log10(num + eps), origin='lower', cmap=cmap); plt.title(f"Numeric log10({name})"); plt.colorbar()
            plt.subplot(1, 3, 2); plt.imshow(jnp.log10(ana + eps), origin='lower', cmap=cmap); plt.title(f"Analytic log10({name})"); plt.colorbar()
            plt.subplot(1, 3, 3); plt.imshow(jnp.log10(rel + eps), origin='lower', cmap='magma'); plt.title("log10(rel. err)"); plt.colorbar()
        else:
            plt.subplot(1, 3, 1); plt.imshow(num, origin='lower', cmap=cmap); plt.title(f"Numeric {name}"); plt.colorbar()
            plt.subplot(1, 3, 2); plt.imshow(ana, origin='lower', cmap=cmap); plt.title(f"Analytic {name}"); plt.colorbar()
            plt.subplot(1, 3, 3); plt.imshow(rel, origin='lower', cmap='magma'); plt.title("rel. err"); plt.colorbar()

        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{name}_{plane}.png")); plt.close()

    # Intensity slices
    slice_and_plot(J_total, J_an, 'x', midx, "J", is_log=True)
    slice_and_plot(J_total, J_an, 'y', midy, "J", is_log=True)
    slice_and_plot(J_total, J_an, 'z', midz, "J", is_log=True)

    # Direction similarity (cosine) slices
    def slice_scalar(arr, plane, idx):
        if plane == 'x':   return arr[idx, 1:-1, 1:-1]
        if plane == 'y':   return arr[1:-1, idx, 1:-1]
        return arr[1:-1, 1:-1, idx]

    for plane, idx in [('x', midx), ('y', midy), ('z', midz)]:
        cs = slice_scalar(cos_sim, plane, idx)
        ang = slice_scalar(angle_err_deg, plane, idx)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.imshow(cs, origin='lower', vmin=0.0, vmax=1.0, cmap='viridis'); plt.title("Cosine similarity"); plt.colorbar()
        plt.subplot(1, 2, 2); plt.imshow(ang, origin='lower', cmap='magma'); plt.title("Angle error (deg)"); plt.colorbar()
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"direction_metrics_{plane}.png")); plt.close()

save_slices()

print(f"Saved figures to: {outdir}")

