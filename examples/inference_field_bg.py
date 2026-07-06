"""
inference_field_bg.py — field-level inference of n_HI(x,y,z) via BACKGROUND
(absorption-tomography) illumination, on a realistic turbulent field.

Motivation
----------
The point-source forward map (inference_field.py / _turb.py) is ~99% degenerate:
flux ~1/r^2, so |dJ/dn_HI| collapses away from the sources and the far field is
unrecoverable. Here the observable is instead a known isotropic background I_bg
absorbed *inward* through the box (ray_trax.background). Sensitivity is ~uniform
(dead_frac ~ 0, sens_dyn_range ~ 2 vs ~1e4), so the FULL box is well-posed.

"Do both":
  1. background illumination     -> well-posed forward map (the main fix)
  2. TV prior + multiscale       -> regularization + conditioning on top

Forward model:  kappa_nu = n_HI * sigma_HI(nu) * DX_PHYS ; J = background(kappa, I_bg)
Optimizer:      Adam on log_n_HI, coarse-to-fine, TV-regularized, full-box loss.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")   # forward is a cumsum; CPU is plenty and dodges GPU contention

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ray_trax.background import compute_background_field_multifreq
from ray_trax.cross_sections import build_kappa_map, NU_HI

# ─── config ──────────────────────────────────────────────────────────────────
GRID        = (64, 64, 64)
N_FREQ      = 6
EPS         = 1e-10
TURB_FILE   = "turbulent_fields/4_s-2.0_rms40p0.npy"
TARGET_MEAN = 1.0
DX_PHYS     = 1e17                       # optical depth per cell = n_HI * sigma * DX_PHYS
LR          = 5e-2
SCALES      = [16, 32, 64]               # coarse-to-fine optimization resolutions
ITERS       = {16: 400, 32: 400, 64: 600}
LAMBDA_TV   = 3e-4                        # TV prior weight (on n_HI)
SAVE_DIR    = "plots_field_bg_inference"
os.makedirs(SAVE_DIR, exist_ok=True)
Nx, Ny, Nz = GRID

nu_bins = jnp.array(np.geomspace(NU_HI, 5.0 * NU_HI, N_FREQ), dtype=jnp.float32)
_zeros  = jnp.zeros(GRID, dtype=jnp.float32)
# background SED (amplitude cancels in the relative loss)
I_bg = ((nu_bins / NU_HI) ** (-1.5)).astype(jnp.float32)
I_bg = I_bg / (I_bg.sum() + EPS)


# ─── true n_HI: realistic turbulent field, block-averaged to GRID, mean-normed ─
def load_turbulent_field(path, shape, target_mean):
    raw = np.load(path).astype(np.float32)
    fx, fy, fz = (s // t for s, t in zip(raw.shape, shape))
    field = raw.reshape(shape[0], fx, shape[1], fy, shape[2], fz).mean(axis=(1, 3, 5))
    field = np.clip(field, 0.0, None)
    field = field / (field.mean() + EPS) * target_mean
    return jnp.asarray(field, dtype=jnp.float32)


n_HI_true = load_turbulent_field(TURB_FILE, GRID, TARGET_MEAN)
print(f"Loaded {TURB_FILE}")
print(f"n_HI true: min={float(n_HI_true.min()):.3f} max={float(n_HI_true.max()):.3f} "
      f"mean={float(n_HI_true.mean()):.3f} std={float(n_HI_true.std()):.3f}")


# ─── forward model ────────────────────────────────────────────────────────────
def forward(n_HI):
    kappa = build_kappa_map(nu_bins, n_HI, _zeros, _zeros) * DX_PHYS
    return compute_background_field_multifreq(kappa, I_bg)


print("\nGenerating J_obs from true n_HI...")
t0 = time.time()
J_obs = forward(n_HI_true)
jax.block_until_ready(J_obs)
print(f"  shape {J_obs.shape}, computed in {time.time()-t0:.3f}s")

# interior mask (drop 1-cell shell)
mask = jnp.ones((*GRID, N_FREQ), dtype=jnp.float32)
for ax in range(3):
    mask = mask.swapaxes(0, ax)
    mask = mask.at[:1].set(0).at[-1:].set(0)
    mask = mask.swapaxes(0, ax)

# sensitivity diagnostic at a uniform reference field
S = jnp.abs(jax.grad(lambda n: jnp.sum(mask * forward(n)))(jnp.ones(GRID, jnp.float32)))
Si = S[1:-1, 1:-1, 1:-1].ravel()
print(f"sensitivity: dyn_range(max/median)={float(Si.max()/(jnp.median(Si)+EPS)):.2f}  "
      f"dead_frac={float((Si < 0.01*Si.max()).mean()):.4f}")


# ─── total variation prior (anisotropic) ──────────────────────────────────────
def tv(field):
    dx = jnp.abs(jnp.diff(field, axis=0)).sum()
    dy = jnp.abs(jnp.diff(field, axis=1)).sum()
    dz = jnp.abs(jnp.diff(field, axis=2)).sum()
    return (dx + dy + dz) / field.size


def upsample(vol, size):
    return jax.image.resize(vol, (size, size, size), method="linear")


# ─── multiscale Adam ──────────────────────────────────────────────────────────
def run_stage(log_n_coarse, scale, n_iter):
    """Optimize log_n at `scale`^3; the forward upsamples it to GRID."""
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(log_n_coarse)

    def loss_fn(log_n):
        log_full = upsample(log_n, Nx) if scale != Nx else log_n
        n_full   = jnp.exp(log_full)
        J_pred   = forward(n_full)
        d        = jnp.log(J_pred + EPS) - jnp.log(J_obs + EPS)
        data     = jnp.sum(mask * d ** 2) / (jnp.sum(mask) + EPS)
        return data + LAMBDA_TV * tv(n_full)

    @jax.jit
    def step(log_n, state):
        loss, g = jax.value_and_grad(loss_fn)(log_n)
        upd, state = optimizer.update(g, state)
        return optax.apply_updates(log_n, upd), state, loss

    losses = []
    for i in range(n_iter):
        log_n_coarse, opt_state, loss = step(log_n_coarse, opt_state)
        losses.append(float(loss))
    return log_n_coarse, losses


print("\nMultiscale optimization (background illumination + TV prior)...")
all_losses = []
log_n = jnp.zeros((SCALES[0],) * 3, dtype=jnp.float32)   # init n_HI = 1
t0 = time.time()
for s in SCALES:
    log_n, losses = run_stage(log_n, s, ITERS[s])
    all_losses += losses
    n_full = jnp.exp(upsample(log_n, Nx) if s != Nx else log_n)
    rel = float(jnp.linalg.norm(n_full - n_HI_true) / (jnp.linalg.norm(n_HI_true) + EPS))
    print(f"  scale {s:2d}^3 | {ITERS[s]} iters | loss={losses[-1]:.3e} | full rel L2={rel:.3e}")
    if s != Nx:
        log_n = upsample(log_n, min([x for x in SCALES if x > s]))
print(f"Done in {time.time()-t0:.1f}s")

n_HI_rec = jnp.exp(upsample(log_n, Nx) if SCALES[-1] != Nx else log_n)
J_rec    = forward(n_HI_rec)

Mi = mask[..., 0]
rel_J = float(jnp.linalg.norm((J_rec - J_obs).ravel()) / (jnp.linalg.norm(J_obs.ravel()) + EPS))
rel_n = float(jnp.linalg.norm((n_HI_rec - n_HI_true).ravel()) / (jnp.linalg.norm(n_HI_true.ravel()) + EPS))
rel_i = float(jnp.linalg.norm((n_HI_rec - n_HI_true) * Mi) / (jnp.linalg.norm(n_HI_true * Mi) + EPS))
print(f"\nFinal rel L2(J):            {rel_J:.3e}")
print(f"Final rel L2(n_HI) interior: {rel_i:.3e}   <-- now the WHOLE interior, no nullspace mask")
print(f"Final rel L2(n_HI) full:     {rel_n:.3e}")

# ─── plots ────────────────────────────────────────────────────────────────────
mid = Nz // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vmin = float(n_HI_true[:, :, mid].min()); vmax = float(n_HI_true[:, :, mid].max())
im = axes[0].imshow(np.array(n_HI_true[:, :, mid]), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title("True $n_{HI}$"); plt.colorbar(im, ax=axes[0])
im = axes[1].imshow(np.array(n_HI_rec[:, :, mid]), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title("Recovered $n_{HI}$"); plt.colorbar(im, ax=axes[1])
im = axes[2].imshow(np.abs(np.array(n_HI_rec[:, :, mid] - n_HI_true[:, :, mid])), origin="lower", cmap="magma")
axes[2].set_title("|Error|"); plt.colorbar(im, ax=axes[2])
plt.suptitle(f"Background-illumination $n_{{HI}}$ inference  |  {os.path.basename(TURB_FILE)}  |  "
             f"full rel L2={rel_n:.2e}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "n_HI_reconstruction.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved n_HI_reconstruction.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(all_losses)
for boundary in np.cumsum([ITERS[s] for s in SCALES])[:-1]:
    ax.axvline(boundary, color="k", ls=":", alpha=0.4)
ax.set_xlabel("Adam iteration (multiscale)"); ax.set_ylabel("loss")
ax.set_title("Background-illumination inference: training curve"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved loss_curve.png")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(np.log10(np.array(S[:, :, mid]) + EPS), origin="lower", cmap="cividis")
ax.set_title("log10 sensitivity $|dJ/dn_{HI}|$\n(uniform, no nullspace)"); plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "sensitivity.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved sensitivity.png")
