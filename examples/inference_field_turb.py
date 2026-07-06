"""
inference_field_turb.py — field-level inference of n_HI(x, y, z) on a REALISTIC
turbulent density field.

Same pipeline as inference_field.py, but the synthetic lognormal truth is replaced
by a precomputed compressible-turbulence density field from turbulent_fields/
(filename encodes power-spectrum slope `s` and density rms). The native field is
128^3; it is block-averaged down to 64^3 and renormalized to mean n_HI = 1 so the
absorption regime (DX_PHYS = 1e17 -> tau_box ~ 4) matches the least-degenerate
sweep config and stays directly comparable to inference_field.py.

Inverse problem: given observed mean intensity J_obs(x, y, z, nu) at multiple
ionizing frequencies, recover the 3D neutral hydrogen density field n_HI(x, y, z).

Forward model:  kappa_nu = n_HI * sigma_HI(nu)  via build_kappa_map
Ray tracer:     compute_radiation_field_multifreq per source, results summed
Optimizer:      Adam (optax)
"""
from autocvd import autocvd
autocvd(num_gpus=1)   # single-GPU: shard_map multi-GPU path crashes under autodiff

import os, time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import random

from ray_trax.ray_trax_3D_multifreq import compute_radiation_field_multifreq
from ray_trax.cross_sections import build_kappa_map, NU_HI

# ─── config ──────────────────────────────────────────────────────────────────
GRID           = (64, 64, 64)
N_FREQ         = 6
N_SOURCES      = 8
NUM_RAYS       = 8000
RAY_BATCH_SIZE = 50
STEP_SIZE      = 0.5
MAX_STEPS      = 320
LR             = 3e-3
N_ITER         = 400
EPS            = 1e-10
F_CUM          = 0.90        # recoverable mask captures this fraction of total sensitivity
TURB_FILE      = "turbulent_fields/4_s-2.0_rms40p0.npy"   # realistic turbulent truth (128^3)
TARGET_MEAN    = 1.0         # renormalize field to this mean n_HI (keeps tau_box ~ 4)
SAVE_DIR       = "plots_field_turb_inference"
os.makedirs(SAVE_DIR, exist_ok=True)
Nx, Ny, Nz = GRID

# Physical size of one grid cell [cm].
# Matches degeneracy_sweep config "dx1e+17": dx_phys=1e17 -> tau_box_med ~ 4,
# the least-degenerate regime found in the sweep.
DX_PHYS = 1e17   # cm per grid unit
# ─────────────────────────────────────────────────────────────────────────────

nu_bins = jnp.array(np.geomspace(NU_HI, 5.0 * NU_HI, N_FREQ), dtype=jnp.float32)

# ─── true n_HI: realistic turbulent field, block-averaged to GRID, mean-normed ─
def load_turbulent_field(path, shape, target_mean):
    """Load a turbulent density field, block-average to `shape`, renormalize mean."""
    raw = np.load(path).astype(np.float32)              # native (e.g. 128^3)
    src = raw.shape
    assert all(s % t == 0 for s, t in zip(src, shape)), \
        f"native shape {src} not divisible by target {shape}"
    fx, fy, fz = (s // t for s, t in zip(src, shape))   # block factors per axis
    # block-average: reshape into (Nx,fx, Ny,fy, Nz,fz) and mean over the block axes
    field = raw.reshape(shape[0], fx, shape[1], fy, shape[2], fz).mean(axis=(1, 3, 5))
    field = np.clip(field, 0.0, None)                   # densities are non-negative
    field = field / (field.mean() + EPS) * target_mean
    return jnp.asarray(field, dtype=jnp.float32)


n_HI_true = load_turbulent_field(TURB_FILE, GRID, TARGET_MEAN)
print(f"Loaded turbulent field: {TURB_FILE}")
print(f"n_HI true (64^3, mean-normed): min={float(n_HI_true.min()):.3f}, "
      f"max={float(n_HI_true.max()):.3f}, mean={float(n_HI_true.mean()):.3f}, "
      f"std={float(n_HI_true.std()):.3f}")

# RNG only used for source placement now (field comes from disk)
key = random.PRNGKey(42)

# ─── sources: 8 random positions, total luminosity = 0.5 ─────────────────────
key, subkey = random.split(key)
# Positions sampled in [0.2, 0.8] × grid extent to keep sources away from walls
source_positions = (random.uniform(subkey, (N_SOURCES, 3)) * 0.6 + 0.2) * jnp.array([Nx, Ny, Nz], dtype=jnp.float32)
source_positions = source_positions.astype(jnp.float32)

print(f"\nSource positions:")
for i, pos in enumerate(np.array(source_positions)):
    print(f"  source {i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

SED = (nu_bins / NU_HI) ** (-1.5)
SED = SED / (SED.sum() + EPS)

x_ = jnp.arange(Nx, dtype=jnp.float32)[:, None, None]
y_ = jnp.arange(Ny, dtype=jnp.float32)[None, :, None]
z_ = jnp.arange(Nz, dtype=jnp.float32)[None, None, :]

# Build j_map for each source: Gaussian blob normalized to sum=1, then scaled
# so that all sources together contribute 0.5 × the original single-source total.
j_maps_list = []
for i in range(N_SOURCES):
    cx = float(source_positions[i, 0])
    cy = float(source_positions[i, 1])
    cz = float(source_positions[i, 2])
    e = jnp.exp(-0.5 * ((x_ - cx)**2 + (y_ - cy)**2 + (z_ - cz)**2) / 2.0**2)
    e = e / (e.sum() + EPS) * (0.5 / N_SOURCES)
    j_maps_list.append(e[..., None] * SED[None, None, None, :])

j_maps = jnp.stack(j_maps_list)   # (N_SOURCES, Nx, Ny, Nz, N_FREQ)

# ─── helpers ─────────────────────────────────────────────────────────────────
_zeros = jnp.zeros(GRID, dtype=jnp.float32)


def forward(n_HI):
    kappa = build_kappa_map(nu_bins, n_HI, _zeros, _zeros) * DX_PHYS
    J = jnp.zeros((*GRID, N_FREQ), dtype=jnp.float32)
    for i in range(N_SOURCES):
        J = J + compute_radiation_field_multifreq(
            j_maps[i], kappa, source_positions[i],
            num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
            ray_batch_size=RAY_BATCH_SIZE, use_sharding=False,
            kappa_interp="trilinear",
        )
    return J


# ─── generate reference J_obs ─────────────────────────────────────────────────
print("\nGenerating J_obs from true n_HI...")
t0    = time.time()
J_obs = forward(n_HI_true)
jax.block_until_ready(J_obs)
print(f"  shape {J_obs.shape}, computed in {time.time()-t0:.2f}s")

# ─── recoverable mask M_rec ───────────────────────────────────────────────────
# Mask out boundary voxels where ray-tracing accuracy is lowest.
mask = jnp.ones((*GRID, N_FREQ), dtype=jnp.float32)
mask = mask.at[:1].set(0);      mask = mask.at[-1:].set(0)
mask = mask.at[:, :1].set(0);   mask = mask.at[:, -1:].set(0)
mask = mask.at[:, :, :1].set(0); mask = mask.at[:, :, -1:].set(0)

# The point-source forward map has a ~99% nullspace (1/r^2 flux dilution): only the
# illuminated core carries an absorption signal. Define a recoverability mask M_rec
# from the forward-map sensitivity S(v) = |d(sum_masked J)/dn_HI(v)|, evaluated at a
# UNIFORM reference field so M_rec depends only on source geometry, not on the truth.
@jax.jit
def _sensitivity(n_ref):
    return jax.grad(lambda n: jnp.sum(mask * forward(n)))(n_ref)

print("\nComputing sensitivity map at uniform reference field...")
t0 = time.time()
S = jnp.abs(_sensitivity(jnp.ones(GRID, dtype=jnp.float32))) * mask[..., 0]
jax.block_until_ready(S)
print(f"  done in {time.time()-t0:.2f}s")

# M_rec = smallest set of voxels capturing F_CUM of total sensitivity.
_S_flat = np.array(S).ravel()
_order  = np.argsort(_S_flat)[::-1]
_csum   = np.cumsum(_S_flat[_order]); _csum /= _csum[-1] + EPS
_n_keep = int(np.searchsorted(_csum, F_CUM) + 1)
_thr    = float(_S_flat[_order[_n_keep - 1]])
M_rec   = (S >= _thr).astype(jnp.float32)            # (Nx,Ny,Nz)
W       = M_rec[..., None] * mask                    # (Nx,Ny,Nz,N_FREQ) weight on J
_frac   = float(M_rec.sum()) / (Nx * Ny * Nz)
print(f"M_rec: f_cum={F_CUM} -> {int(M_rec.sum())} voxels ({100*_frac:.2f}% of box)")


# ─── loss: log-space data term, restricted to M_rec ───────────────────────────
# Working in log(J) linearizes the multiplicative attenuation (I = I0*exp(-tau),
# tau linear in n_HI) and compresses the heavy-tailed 1/r^2 dynamic range; the W
# weight confines the fit to the illuminated core where J actually depends on n_HI,
# undoing the 1/(J+eps) over-weighting of dark/nullspace voxels.
def loss_fn(log_n_HI):
    J_pred = forward(jnp.exp(log_n_HI))
    d = jnp.log(J_pred + EPS) - jnp.log(J_obs + EPS)
    return jnp.sum(W * d ** 2) / (jnp.sum(W) + EPS)


# ─── Adam optimizer ───────────────────────────────────────────────────────────
optimizer = optax.adam(LR)
log_n_HI  = jnp.zeros(GRID, dtype=jnp.float32)   # init: n_HI = exp(0) = 1
opt_state = optimizer.init(log_n_HI)


@jax.jit
def step_fn(log_n_HI, opt_state):
    loss, grads        = jax.value_and_grad(loss_fn)(log_n_HI)
    updates, new_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(log_n_HI, updates), new_state, loss


# ─── optimization loop ────────────────────────────────────────────────────────
print(f"\nRunning {N_ITER} Adam steps  (lr={LR}, grid {Nx}x{Ny}x{Nz}, "
      f"{N_FREQ} freq bins, {N_SOURCES} sources)...")
losses = []
t0 = time.time()
for i in range(N_ITER):
    log_n_HI, opt_state, loss_val = step_fn(log_n_HI, opt_state)
    losses.append(float(loss_val))
    if i % 50 == 0 or i == N_ITER - 1:
        n_cur   = jnp.exp(log_n_HI)
        rel_rec = float(jnp.linalg.norm((n_cur - n_HI_true) * M_rec)
                        / (jnp.linalg.norm(n_HI_true * M_rec) + EPS))
        print(f"  iter {i:4d} | loss={float(loss_val):.3e} | rel_err(M_rec)={rel_rec:.3e}")

print(f"Done in {time.time()-t0:.1f}s")

# ─── final evaluation ─────────────────────────────────────────────────────────
n_HI_rec = jnp.exp(log_n_HI)
J_rec    = forward(n_HI_rec)
jax.block_until_ready(J_rec)

M_comp = (1.0 - M_rec) * mask[..., 0]   # recoverable complement (interior only)
rel_J     = float(jnp.linalg.norm((J_rec - J_obs).ravel()) / (jnp.linalg.norm(J_obs.ravel()) + EPS))
rel_n     = float(jnp.linalg.norm((n_HI_rec - n_HI_true).ravel()) / (jnp.linalg.norm(n_HI_true.ravel()) + EPS))
rel_n_rec = float(jnp.linalg.norm((n_HI_rec - n_HI_true) * M_rec)
                  / (jnp.linalg.norm(n_HI_true * M_rec) + EPS))
rel_n_cmp = float(jnp.linalg.norm((n_HI_rec - n_HI_true) * M_comp)
                  / (jnp.linalg.norm(n_HI_true * M_comp) + EPS))
print(f"\nFinal rel L2(J):           {rel_J:.3e}")
print(f"Final rel L2(n_HI) on M_rec: {rel_n_rec:.3e}   <-- well-posed target")
print(f"Final rel L2(n_HI) complement: {rel_n_cmp:.3e}   (nullspace, not constrained)")
print(f"Final rel L2(n_HI) full:    {rel_n:.3e}")

# ─── plots ────────────────────────────────────────────────────────────────────
mid = Nz // 2

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
vmin = float(n_HI_true[:, :, mid].min())
vmax = float(n_HI_true[:, :, mid].max())

im = axes[0].imshow(np.array(n_HI_true[:, :, mid]), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
axes[0].set_title("True $n_{HI}$"); plt.colorbar(im, ax=axes[0])

im = axes[1].imshow(np.array(n_HI_rec[:, :, mid]), origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
axes[1].set_title("Recovered $n_{HI}$"); plt.colorbar(im, ax=axes[1])

err = np.abs(np.array(n_HI_rec[:, :, mid] - n_HI_true[:, :, mid]))
im = axes[2].imshow(err, origin='lower', cmap='magma')
axes[2].set_title("|Error|"); plt.colorbar(im, ax=axes[2])

# Mark source positions on each panel
for ax in axes:
    src = np.array(source_positions)
    in_slice = np.abs(src[:, 2] - mid) < 4
    ax.scatter(src[in_slice, 1], src[in_slice, 0], c='white', marker='+',
               s=60, linewidths=1.5, label='sources (near slice)')

plt.suptitle(f"Turbulent $n_{{HI}}$ inference  |  {os.path.basename(TURB_FILE)}  |  "
             f"{N_SOURCES} sources  |  rel L2={rel_n:.2e}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "n_HI_reconstruction.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved n_HI_reconstruction.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(losses)
ax.set_xlabel("Adam iteration"); ax.set_ylabel("Relative L2 loss")
ax.set_title(f"Turbulent-field inference: training curve ({N_SOURCES} sources)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved loss_curve.png")

fidx = N_FREQ // 2
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
lo = float(jnp.log10(J_obs[:, :, mid, fidx] + EPS).min())
hi = float(jnp.log10(J_obs[:, :, mid, fidx] + EPS).max())

im = axes[0].imshow(np.log10(np.array(J_obs[:, :, mid, fidx]) + EPS),
                    origin='lower', cmap='inferno', vmin=lo, vmax=hi)
axes[0].set_title(f"log10 J_obs  (ν bin {fidx})"); plt.colorbar(im, ax=axes[0])

im = axes[1].imshow(np.log10(np.array(J_rec[:, :, mid, fidx]) + EPS),
                    origin='lower', cmap='inferno', vmin=lo, vmax=hi)
axes[1].set_title(f"log10 J_rec  (ν bin {fidx})"); plt.colorbar(im, ax=axes[1])

plt.suptitle(f"Radiation field comparison  |  rel L2(J)={rel_J:.2e}", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "J_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved J_comparison.png")
