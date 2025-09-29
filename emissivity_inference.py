#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main: emissivity field reconstruction via differentiable forward operator.

Structure mirrors example_3D_time_dep_turb.py:
1) Config
2) Load fields
3) Build sources
4) Make / load measurements
5) Invert (TV+L2 regularized, non-negativity)
6) Save quicklooks
"""

import os, time, gc
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- import your library pieces (you provide these) ---
# forward_measurements: E, kappa, sources -> per-ray integrals (JAX-diffable)
# reconstruct_emissivity: gradient-based MAP with TV/L2 priors (returns E_est, losses)
# InverseConfig: dataclass with (lambda_tv, lambda_l2, lr, steps, clip_norm)
from ray_trax.ray_trax_3D_tdep_experimental import forward_measurements, reconstruct_emissivity, InverseConfig
from ray_trax.utils import process_density_field   # same helper you use in the example

# ----------------------------
# Config (edit as needed)
# ----------------------------
outdir = "plots_emissivity_inference"
os.makedirs(outdir, exist_ok=True)

# Physics / grid
c = 1.0
dt = 1.0
step_size = 0.5
num_rays = 4096
seed = 42

# Data
path_kappa = "./turbulent_fields/0_s-1.7_rms20p0.npy"     # absorption ∝ density
path_y_obs = None   # if provided: npy with shape [S * num_rays]

# Sources
num_sources = 16        # pick top-N peaks from density via your helper

# Inverse hyperparams
cfg = InverseConfig(lambda_tv=1e-3, lambda_l2=1e-6, lr=1e-2, steps=2000, clip_norm=1.0)

# ----------------------------
# Load fields & pick sources
# ----------------------------
print("[i] Loading kappa/density ...")
kappa_base = jnp.array(np.load(path_kappa))      # [Nx,Ny,Nz]
emissivity_seed, mask, star_positions = process_density_field(kappa_base, percentile=99.99)
# We only use star_positions here; emissivity_seed is just for plotting/reference.

S = star_positions.shape[0]
if S > num_sources:
    star_positions = star_positions[:num_sources]
    S = num_sources

print(f"[i] Using {S} sources, {num_rays} rays/source.")

# ----------------------------
# Measurements (y_obs)
# ----------------------------
key = jax.random.PRNGKey(seed)

n_steps = max(1, int(np.floor(float(c) * float(dt) / float(step_size))))

if path_y_obs is None:
    print("[i] No observed measurements provided; generating synthetic y_obs from a faint E_true.")
    # Synthetic faint emissivity for a reasonable inverse test
    E_true = 0.1 * emissivity_seed / (jnp.max(emissivity_seed) + 1e-12)
    y_obs = forward_measurements(
        emissivity=E_true,
        kappa=kappa_base,
        star_positions=star_positions,
        num_rays=num_rays,
        step_size=step_size,
        c=c,
        dt=dt,
        key=key,
        use_sharded=False,    # flip True if you wired sharding
        n_steps=n_steps
    )
else:
    print(f"[i] Loading y_obs from {path_y_obs}")
    y_obs = jnp.array(np.load(path_y_obs))

# Optional normalization (helps optimization)
scale = jnp.median(jnp.abs(y_obs)) + 1e-12
y_obs_n = y_obs / scale

# ----------------------------
# Inversion
# ----------------------------
print("[i] Starting emissivity reconstruction ...")
t0 = time.time()
E_est_n, losses = reconstruct_emissivity(
    y_obs=y_obs_n,
    kappa=kappa_base,
    star_positions=star_positions,
    num_rays=num_rays,
    step_size=step_size,
    c=c,
    dt=dt,
    cfg=cfg,
    key=key,
    init_emissivity=None,   # or pass a warm start (same shape as kappa)
    use_sharded=False,
)
# rescale back
E_est = E_est_n * scale
t1 = time.time()
print(f"[✓] Done in {t1 - t0:.1f}s")

# ----------------------------
# Save quicklooks
# ----------------------------
Nx, Ny, Nz = kappa_base.shape
midz = Nz // 2

def save_slice(img, title, fname):
    plt.figure(figsize=(6,5))
    plt.imshow(jnp.log10(img[:, :, midz] + 1e-8), origin="lower", cmap="inferno")
    plt.colorbar(label="log10")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close()

print("[i] Writing quicklook slices ...")
save_slice(kappa_base, "kappa (mid-Z)", "kappa_midZ.png")
save_slice(E_est, "E_est (mid-Z)", "E_est_midZ.png")
if path_y_obs is None:
    save_slice(E_true, "E_true (mid-Z)", "E_true_midZ.png")

# loss curve
import numpy as np
L = np.array([float(l[0]) for l in losses])
Ld = np.array([float(l[1]) for l in losses])
Lr = np.array([float(l[2]) for l in losses])
plt.figure(figsize=(6,4))
plt.plot(L, label="total")
plt.plot(Ld, label="data")
plt.plot(Lr, label="reg")
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "loss_curves.png"), dpi=150)
plt.close()

# save volumes
np.save(os.path.join(outdir, "E_est.npy"), np.array(E_est))
if path_y_obs is None:
    np.save(os.path.join(outdir, "E_true.npy"), np.array(E_true))
np.save(os.path.join(outdir, "kappa.npy"), np.array(kappa_base))

# clean up
del kappa_base, E_est
gc.collect()
print("[✓] Results in:", outdir)

