"""
inference_field_multispecies.py — multi-species point-source field inference.

Recovers THREE density fields (n_HI, n_HeI, n_HeII) simultaneously from a single
multi-frequency radiation field J_obs(x,y,z,nu), using the point-source ray tracer.

Why this exploits the "full multi-frequency potential"
------------------------------------------------------
With only HI, every frequency bin measures the same field n_HI times a known
function sigma_HI(nu) -> the frequency axis is redundant. Turning on HeI/HeII makes
frequency genuinely informative, because the three cross sections have different
thresholds and slopes:
    nu < 24.6 eV      : only HI absorbs
    24.6 - 54.4 eV    : HI + HeI
    nu > 54.4 eV      : HI + HeI + HeII
so multi-frequency J disentangles the three maps. The bin grid geomspace(NU_HI,
5*NU_HI) = 13.6 -> 68 eV already straddles all three thresholds.

Caveat: this does NOT remove the point-source 1/r^2 geometric nullspace (dark far
field is dark for every species). Multi-species improves the *illuminated-region*
inversion and adds real spectral separation; it is not a substitute for changing the
illumination geometry. Recovery is therefore reported on the geometry mask M_rec.

Forward: kappa_nu = (n_HI sigma_HI + n_HeI sigma_HeI + n_HeII sigma_HeII) * DX_PHYS
         J = sum_sources compute_radiation_field_multifreq(..., kappa_interp="trilinear")
Optimizer: Adam on (log n_HI, log n_HeI, log n_HeII) jointly, log-space masked L2.
"""
from autocvd import autocvd
autocvd(num_gpus=1)

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
NUM_RAYS       = int(os.environ.get("NUM_RAYS", 8000))
RAY_BATCH_SIZE = 50
STEP_SIZE      = 0.5
MAX_STEPS      = int(os.environ.get("MAX_STEPS", 320))
LR             = 3e-3
N_ITER         = int(os.environ.get("N_ITER", 400))
EPS            = 1e-10
F_CUM          = 0.90
DX_PHYS        = float(os.environ.get("DX_PHYS", 1e17))   # cm/cell; lower -> lower optical depth

# Three independent turbulent fields (different slope/rms) give the species distinct
# spatial structure -> a genuine disentanglement test. Each is mean-normalized then
# scaled to a mean abundance. Cosmic He/H ~ 0.08; the He means here are inflated so
# both He species leave a detectable multi-frequency imprint (tunable).
SPECIES   = ("HI", "HeI", "HeII")
TURB_FILE = {"HI": "turbulent_fields/4_s-2.0_rms40p0.npy",
             "HeI": "turbulent_fields/1_s-1.7_rms40p0.npy",
             "HeII": "turbulent_fields/7_s-2.3_rms40p0.npy"}
ABUND     = {"HI": 1.0, "HeI": 0.3, "HeII": 0.3}   # mean of each field

SAVE_DIR  = os.environ.get("SAVE_DIR", "plots_field_multispecies_inference")
os.makedirs(SAVE_DIR, exist_ok=True)
Nx, Ny, Nz = GRID
nu_bins = jnp.array(np.geomspace(NU_HI, 5.0 * NU_HI, N_FREQ), dtype=jnp.float32)


# ─── truth fields ─────────────────────────────────────────────────────────────
def load_turbulent_field(path, shape, mean_val):
    raw = np.load(path).astype(np.float32)
    fx, fy, fz = (s // t for s, t in zip(raw.shape, shape))
    field = raw.reshape(shape[0], fx, shape[1], fy, shape[2], fz).mean(axis=(1, 3, 5))
    field = np.clip(field, 0.0, None)
    field = field / (field.mean() + EPS) * mean_val
    return jnp.asarray(field, dtype=jnp.float32)


n_true = {s: load_turbulent_field(TURB_FILE[s], GRID, ABUND[s]) for s in SPECIES}
for s in SPECIES:
    f = n_true[s]
    print(f"n_{s:4s} true: min={float(f.min()):.3f} max={float(f.max()):.3f} "
          f"mean={float(f.mean()):.3f} std={float(f.std()):.3f}")

# ─── sources (same as single-species point-source setup) ──────────────────────
key = random.PRNGKey(42)
key, subkey = random.split(key)
source_positions = ((random.uniform(subkey, (N_SOURCES, 3)) * 0.6 + 0.2)
                    * jnp.array([Nx, Ny, Nz], dtype=jnp.float32)).astype(jnp.float32)

SED = (nu_bins / NU_HI) ** (-1.5)
SED = SED / (SED.sum() + EPS)

x_ = jnp.arange(Nx, dtype=jnp.float32)[:, None, None]
y_ = jnp.arange(Ny, dtype=jnp.float32)[None, :, None]
z_ = jnp.arange(Nz, dtype=jnp.float32)[None, None, :]
j_maps_list = []
for i in range(N_SOURCES):
    cx, cy, cz = (float(source_positions[i, k]) for k in range(3))
    e = jnp.exp(-0.5 * ((x_ - cx)**2 + (y_ - cy)**2 + (z_ - cz)**2) / 2.0**2)
    e = e / (e.sum() + EPS) * (0.5 / N_SOURCES)
    j_maps_list.append(e[..., None] * SED[None, None, None, :])
j_maps = jnp.stack(j_maps_list)


# ─── forward model (all three species) ────────────────────────────────────────
def forward(n_HI, n_HeI, n_HeII):
    kappa = build_kappa_map(nu_bins, n_HI, n_HeI, n_HeII) * DX_PHYS
    J = jnp.zeros((*GRID, N_FREQ), dtype=jnp.float32)
    for i in range(N_SOURCES):
        J = J + compute_radiation_field_multifreq(
            j_maps[i], kappa, source_positions[i],
            num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
            ray_batch_size=RAY_BATCH_SIZE, use_sharding=False,
            kappa_interp="trilinear",
        )
    return J


print("\nGenerating J_obs from true fields...")
t0 = time.time()
J_obs = forward(n_true["HI"], n_true["HeI"], n_true["HeII"])
jax.block_until_ready(J_obs)
print(f"  shape {J_obs.shape}, computed in {time.time()-t0:.2f}s")

# ─── geometry mask M_rec (from n_HI sensitivity at a uniform reference) ────────
mask = jnp.ones((*GRID, N_FREQ), dtype=jnp.float32)
mask = mask.at[:1].set(0).at[-1:].set(0)
mask = mask.at[:, :1].set(0).at[:, -1:].set(0)
mask = mask.at[:, :, :1].set(0).at[:, :, -1:].set(0)

ref = {s: jnp.full(GRID, ABUND[s], jnp.float32) for s in SPECIES}


@jax.jit
def _sensitivity():
    return jax.grad(lambda n: jnp.sum(mask * forward(n, ref["HeI"], ref["HeII"])))(ref["HI"])

print("\nComputing sensitivity map (geometry, at uniform reference)...")
t0 = time.time()
S = jnp.abs(_sensitivity()) * mask[..., 0]
jax.block_until_ready(S)
print(f"  done in {time.time()-t0:.2f}s")

_S = np.array(S).ravel(); _o = np.argsort(_S)[::-1]
_c = np.cumsum(_S[_o]); _c /= _c[-1] + EPS
_thr = float(_S[_o[int(np.searchsorted(_c, F_CUM))]])
M_rec = (S >= _thr).astype(jnp.float32)
W = M_rec[..., None] * mask
print(f"M_rec: {int(M_rec.sum())} voxels ({100*float(M_rec.sum())/(Nx*Ny*Nz):.2f}% of box)")


# ─── joint loss over the three log-fields ─────────────────────────────────────
# Init at log(mean abundance): mean abundances assumed known, recover the fluctuations.
params = {s: jnp.full(GRID, np.log(ABUND[s]), jnp.float32) for s in SPECIES}


def loss_fn(p):
    J_pred = forward(jnp.exp(p["HI"]), jnp.exp(p["HeI"]), jnp.exp(p["HeII"]))
    d = jnp.log(J_pred + EPS) - jnp.log(J_obs + EPS)
    return jnp.sum(W * d ** 2) / (jnp.sum(W) + EPS)


optimizer = optax.adam(LR)
opt_state = optimizer.init(params)


@jax.jit
def step_fn(p, state):
    loss, g = jax.value_and_grad(loss_fn)(p)
    upd, state = optimizer.update(g, state)
    return optax.apply_updates(p, upd), state, loss


def rel_on(p, s, region):
    return float(jnp.linalg.norm((jnp.exp(p[s]) - n_true[s]) * region)
                 / (jnp.linalg.norm(n_true[s] * region) + EPS))


print(f"\nRunning {N_ITER} Adam steps (joint HI/HeI/HeII, {N_SOURCES} sources)...")
losses = []
t0 = time.time()
for i in range(N_ITER):
    params, opt_state, loss_val = step_fn(params, opt_state)
    losses.append(float(loss_val))
    if i % 50 == 0 or i == N_ITER - 1:
        errs = "  ".join(f"{s}={rel_on(params, s, M_rec):.3f}" for s in SPECIES)
        print(f"  iter {i:4d} | loss={float(loss_val):.3e} | relL2(M_rec) {errs}")
    # periodic checkpoint so a long run survives a crash (latest state only)
    if i % 100 == 0 or i == N_ITER - 1:
        np.savez(os.path.join(SAVE_DIR, "checkpoint.npz"),
                 iteration=i, losses=np.array(losses),
                 **{s: np.array(jnp.exp(params[s])) for s in SPECIES})
print(f"Done in {time.time()-t0:.1f}s")

# ─── final eval ───────────────────────────────────────────────────────────────
Mi = mask[..., 0]
print("\nFinal relative L2 per species:")
for s in SPECIES:
    print(f"  n_{s:4s}: M_rec={rel_on(params, s, M_rec):.3e}   "
          f"full={rel_on(params, s, jnp.ones(GRID)):.3e}")

# reconstructed intensity field (for the J plot below)
J_rec = forward(jnp.exp(params["HI"]), jnp.exp(params["HeI"]), jnp.exp(params["HeII"]))
jax.block_until_ready(J_rec)
rel_J = float(jnp.linalg.norm((J_rec - J_obs).ravel()) / (jnp.linalg.norm(J_obs.ravel()) + EPS))
print(f"Final rel L2(J): {rel_J:.3e}")

# ─── plots: one row per species (true / recovered / |error|) ──────────────────
mid = Nz // 2
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
for r, s in enumerate(SPECIES):
    t = np.array(n_true[s][:, :, mid]); rec = np.array(jnp.exp(params[s])[:, :, mid])
    vmin, vmax = float(t.min()), float(t.max())
    im = axes[r, 0].imshow(t, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[r, 0].set_ylabel(f"n_{s}", fontsize=12); axes[r, 0].set_title("true" if r == 0 else "")
    plt.colorbar(im, ax=axes[r, 0])
    im = axes[r, 1].imshow(rec, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[r, 1].set_title("recovered" if r == 0 else ""); plt.colorbar(im, ax=axes[r, 1])
    im = axes[r, 2].imshow(np.abs(rec - t), origin="lower", cmap="magma")
    axes[r, 2].set_title("|error|" if r == 0 else ""); plt.colorbar(im, ax=axes[r, 2])
plt.suptitle("Multi-species point-source inference (HI / HeI / HeII)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "multispecies_reconstruction.png"), dpi=160, bbox_inches="tight")
plt.close(); print("Saved multispecies_reconstruction.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(losses); ax.set_xlabel("Adam iteration"); ax.set_ylabel("masked log-L2 loss")
ax.set_title("Multi-species inference: training curve"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close(); print("Saved loss_curve.png")

# ─── reconstructed intensity field J (obs vs recovered) ───────────────────────
# One panel per frequency bin: log10 J_obs (top) vs log10 J_rec (bottom), shared scale.
fig, axes = plt.subplots(2, N_FREQ, figsize=(3 * N_FREQ, 6))
for f in range(N_FREQ):
    lo = float(jnp.log10(J_obs[:, :, mid, f] + EPS).min())
    hi = float(jnp.log10(J_obs[:, :, mid, f] + EPS).max())
    a0 = axes[0, f].imshow(np.log10(np.array(J_obs[:, :, mid, f]) + EPS),
                           origin="lower", cmap="inferno", vmin=lo, vmax=hi)
    axes[0, f].set_title(f"bin {f}\n{float(nu_bins[f]/NU_HI):.1f} $\\nu_{{HI}}$", fontsize=9)
    a1 = axes[1, f].imshow(np.log10(np.array(J_rec[:, :, mid, f]) + EPS),
                           origin="lower", cmap="inferno", vmin=lo, vmax=hi)
    if f == 0:
        axes[0, f].set_ylabel("log10 J_obs"); axes[1, f].set_ylabel("log10 J_rec")
plt.suptitle(f"Reconstructed intensity field  |  rel L2(J)={rel_J:.2e}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "J_reconstruction.png"), dpi=160, bbox_inches="tight")
plt.close(); print("Saved J_reconstruction.png")
