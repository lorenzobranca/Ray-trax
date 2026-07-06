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

# ---------- Direction plotting utilities ----------
import numpy as np
import matplotlib.pyplot as plt
import os

def _slice2d(arr3d, plane, idx):
    """Return a 2D slice from a 3D array with 'ij' indexing."""
    if plane == 'x':
        return arr3d[idx, :, :]
    elif plane == 'y':
        return arr3d[:, idx, :]
    elif plane == 'z':
        return arr3d[:, :, idx]
    else:
        raise ValueError("plane must be 'x', 'y', or 'z'.")

def _slice2d_vec(arr4d, plane, idx, comp):
    """Return a 2D slice from a 4D vector field (Nx,Ny,Nz,3)."""
    return _slice2d(arr4d[..., comp], plane, idx)

def plot_direction_components_triptych(
    dir_num, dir_an, plane, idx, outdir, fname="direction_components.png"
):
    """
    3x3 figure like intensity: rows = (x,y,z), cols = (numeric, analytic, |diff|).
    Components are in [-1,1]; diffs in [0,1].
    """
    os.makedirs(outdir, exist_ok=True)
    comps = ['x', 'y', 'z']
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)

    # Fixed color ranges for clarity
    vmin_c, vmax_c = -1.0, 1.0
    vmin_d, vmax_d = 0.0, 1.0

    im_cbars = [None, None, None]  # to share column colorbars

    for r, comp in enumerate(comps):
        c = {'x':0, 'y':1, 'z':2}[comp]

        num = np.asarray(_slice2d_vec(dir_num, plane, idx, c))
        ana = np.asarray(_slice2d_vec(dir_an, plane, idx, c))
        diff = np.abs(num - ana)

        # Numeric
        im0 = axes[r, 0].imshow(num, origin='lower', cmap='coolwarm',
                                vmin=vmin_c, vmax=vmax_c)
        axes[r, 0].set_title(f"{comp}-component (numeric)")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if im_cbars[0] is None:
            im_cbars[0] = im0

        # Analytic
        im1 = axes[r, 1].imshow(ana, origin='lower', cmap='coolwarm',
                                vmin=vmin_c, vmax=vmax_c)
        axes[r, 1].set_title(f"{comp}-component (analytic)")
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        if im_cbars[1] is None:
            im_cbars[1] = im1

        # |diff|
        im2 = axes[r, 2].imshow(diff, origin='lower', cmap='magma',
                                vmin=vmin_d, vmax=vmax_d)
        axes[r, 2].set_title(f"{comp}-component |diff|")
        axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
        if im_cbars[2] is None:
            im_cbars[2] = im2

    # One colorbar per column
    for col, im in enumerate(im_cbars):
        cax = fig.add_axes([0.92, 0.70 - 0.31*col, 0.015, 0.25])  # right-side bars
        fig.colorbar(im, cax=cax)

    fig.suptitle(f"Direction components — plane {plane}, slice {idx}", y=0.98)
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_direction_metrics(
    dir_num, dir_an, J_num=None, min_J=None,
    plane='z', idx=0, outdir='.', fname='direction_metrics.png', eps=1e-10
):
    """
    1x2 figure: cosine similarity (0..1) and angle error (deg).
    Optionally masks low-intensity pixels via J_num < min_J.
    """
    os.makedirs(outdir, exist_ok=True)

    # Cosine similarity & angle error
    dot = np.sum(np.asarray(dir_num) * np.asarray(dir_an), axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    cos_sim = dot
    ang_err = np.degrees(np.arccos(cos_sim + 0.0))

    if (J_num is not None) and (min_J is not None):
        mask = (np.asarray(J_num) < float(min_J))
        cos_sim = np.where(mask, np.nan, cos_sim)
        ang_err = np.where(mask, np.nan, ang_err)

    cs2d  = _slice2d(cos_sim, plane, idx)
    ang2d = _slice2d(ang_err, plane, idx)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im0 = axes[0].imshow(cs2d, origin='lower', vmin=0.0, vmax=1.0, cmap='viridis')
    axes[0].set_title("Cosine similarity")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0])

    # Pick a robust upper bound for angles: 99th percentile (cap at 60 deg to keep scale readable)
    finite_ang = ang2d[np.isfinite(ang2d)]
    if finite_ang.size:
        vmax_ang = float(np.minimum(60.0, np.percentile(finite_ang, 99)))
    else:
        vmax_ang = 30.0

    im1 = axes[1].imshow(ang2d, origin='lower', cmap='magma', vmin=0.0, vmax=vmax_ang)
    axes[1].set_title("Angle error (deg)")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1])

    fig.suptitle(f"Direction metrics — plane {plane}, slice {idx}", y=1.02)
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def save_dir_plots(dir_num, dir_an, J_num=None, min_J=None, outdir='.', eps=1e-10):
    """
    Save direction visuals for central x/y/z slices:
      - 3x3 component triptychs (numeric | analytic | |diff|)
      - 1x2 metrics (cosine similarity | angle error)
    """
    Nx, Ny, Nz, _ = dir_num.shape
    centers = {'x': Nx//2, 'y': Ny//2, 'z': Nz//2}

    for plane, idx in centers.items():
        plot_direction_components_triptych(
            dir_num, dir_an, plane, idx, outdir,
            fname=f"direction_components_{plane}.png"
        )
        plot_direction_metrics(
            dir_num, dir_an, J_num=J_num, min_J=min_J,
            plane=plane, idx=idx, outdir=outdir,
            fname=f"direction_metrics_{plane}.png", eps=eps
        )
save_slices()
min_J_for_metrics = np.percentile(np.asarray(J_total), 60)
save_dir_plots(dir_total, dir_an, J_num=J_total, min_J=min_J_for_metrics, outdir=outdir)

print(f"Saved figures to: {outdir}")

