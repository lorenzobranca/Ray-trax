"""
opacity_sweep.py — (1) regenerate the reconstructed-J plot for the finished
multi-species run from its checkpoint, and (2) sweep DX_PHYS (average opacity)
measuring the recoverable fraction vs the absorption signal, WITHOUT re-optimizing.

The point is to test the "lower the average opacity" idea cheaply: everything
here is forward + one sensitivity gradient per DX (seconds each), no Adam loop.
"""
from autocvd import autocvd
autocvd(num_gpus=1)

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import random

from ray_trax.ray_trax_3D_multifreq import compute_radiation_field_multifreq
from ray_trax.cross_sections import build_kappa_map, NU_HI

GRID       = (64, 64, 64)
N_FREQ     = 6
N_SOURCES  = 8
STEP_SIZE  = 0.5
MAX_STEPS  = 320
EPS        = 1e-10
F_CUM      = 0.90
SPECIES    = ("HI", "HeI", "HeII")
TURB_FILE  = {"HI": "turbulent_fields/4_s-2.0_rms40p0.npy",
              "HeI": "turbulent_fields/1_s-1.7_rms40p0.npy",
              "HeII": "turbulent_fields/7_s-2.3_rms40p0.npy"}
ABUND      = {"HI": 1.0, "HeI": 0.3, "HeII": 0.3}
SAVE_DIR   = "plots_field_multispecies_inference"
Nx, Ny, Nz = GRID
nu_bins = jnp.array(np.geomspace(NU_HI, 5.0 * NU_HI, N_FREQ), dtype=jnp.float32)


def load_field(path, mean_val):
    raw = np.load(path).astype(np.float32)
    fx, fy, fz = (s // t for s, t in zip(raw.shape, GRID))
    f = raw.reshape(Nx, fx, Ny, fy, Nz, fz).mean(axis=(1, 3, 5))
    f = np.clip(f, 0.0, None); f = f / (f.mean() + EPS) * mean_val
    return jnp.asarray(f, dtype=jnp.float32)


n_true = {s: load_field(TURB_FILE[s], ABUND[s]) for s in SPECIES}
ref    = {s: jnp.full(GRID, ABUND[s], jnp.float32) for s in SPECIES}

# sources / j_maps (identical to the inference script)
key = random.PRNGKey(42); key, sub = random.split(key)
source_positions = ((random.uniform(sub, (N_SOURCES, 3)) * 0.6 + 0.2)
                    * jnp.array([Nx, Ny, Nz], jnp.float32)).astype(jnp.float32)
SED = (nu_bins / NU_HI) ** (-1.5); SED = SED / (SED.sum() + EPS)
x_ = jnp.arange(Nx, dtype=jnp.float32)[:, None, None]
y_ = jnp.arange(Ny, dtype=jnp.float32)[None, :, None]
z_ = jnp.arange(Nz, dtype=jnp.float32)[None, None, :]
jm = []
for i in range(N_SOURCES):
    cx, cy, cz = (float(source_positions[i, k]) for k in range(3))
    e = jnp.exp(-0.5 * ((x_ - cx)**2 + (y_ - cy)**2 + (z_ - cz)**2) / 4.0)
    e = e / (e.sum() + EPS) * (0.5 / N_SOURCES)
    jm.append(e[..., None] * SED[None, None, None, :])
j_maps = jnp.stack(jm)

mask = jnp.ones((*GRID, N_FREQ), jnp.float32)
mask = mask.at[:1].set(0).at[-1:].set(0).at[:, :1].set(0).at[:, -1:].set(0).at[:, :, :1].set(0).at[:, :, -1:].set(0)


def forward(nHI, nHeI, nHeII, DX, num_rays):
    kappa = build_kappa_map(nu_bins, nHI, nHeI, nHeII) * DX
    J = jnp.zeros((*GRID, N_FREQ), jnp.float32)
    for i in range(N_SOURCES):
        J = J + compute_radiation_field_multifreq(
            j_maps[i], kappa, source_positions[i], num_rays=num_rays,
            step_size=STEP_SIZE, max_steps=MAX_STEPS, ray_batch_size=50,
            use_sharding=False, kappa_interp="trilinear")
    return J


# ══ Part 1: regenerate J plot from checkpoint (production DX, 8000 rays) ══════
print("Regenerating J_reconstruction.png from checkpoint...")
ck = np.load(os.path.join(SAVE_DIR, "checkpoint.npz"))
rec = {s: jnp.asarray(ck[s], jnp.float32) for s in SPECIES}
DX_PROD, NR = 1e17, 8000
J_obs = forward(n_true["HI"], n_true["HeI"], n_true["HeII"], DX_PROD, NR)
J_rec = forward(rec["HI"], rec["HeI"], rec["HeII"], DX_PROD, NR)
rel_J = float(jnp.linalg.norm((J_rec - J_obs).ravel()) / (jnp.linalg.norm(J_obs.ravel()) + EPS))
mid = Nz // 2
fig, axes = plt.subplots(2, N_FREQ, figsize=(3 * N_FREQ, 6))
for f in range(N_FREQ):
    lo = float(jnp.log10(J_obs[:, :, mid, f] + EPS).min())
    hi = float(jnp.log10(J_obs[:, :, mid, f] + EPS).max())
    axes[0, f].imshow(np.log10(np.array(J_obs[:, :, mid, f]) + EPS), origin="lower",
                      cmap="inferno", vmin=lo, vmax=hi)
    axes[0, f].set_title(f"bin {f}\n{float(nu_bins[f]/NU_HI):.1f}$\\nu_{{HI}}$", fontsize=9)
    axes[1, f].imshow(np.log10(np.array(J_rec[:, :, mid, f]) + EPS), origin="lower",
                      cmap="inferno", vmin=lo, vmax=hi)
    if f == 0:
        axes[0, f].set_ylabel("log10 J_obs"); axes[1, f].set_ylabel("log10 J_rec")
plt.suptitle(f"Reconstructed intensity field (from checkpoint)  |  rel L2(J)={rel_J:.2e}", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "J_reconstruction.png"), dpi=160, bbox_inches="tight")
plt.close()
print(f"  saved J_reconstruction.png  (rel L2(J)={rel_J:.3e})")


# ══ Part 2: opacity sweep (diagnostics only, 2000 rays) ══════════════════════
DX_LIST = [1e17, 3e16, 1e16, 3e15, 1e15]
NR_SW = 2000
Mi = mask[..., 0]


def sensitivity(DX):
    return jnp.abs(jax.grad(
        lambda n: jnp.sum(mask * forward(n, ref["HeI"], ref["HeII"], DX, NR_SW)))(ref["HI"])) * Mi


print(f"\nOpacity sweep (NUM_RAYS={NR_SW}):")
print(f"{'DX_PHYS':>10} {'tau_box_med':>12} {'M_rec_frac':>11} {'dead_frac':>10} {'dJ_rel':>9}")
rows = []
for DX in DX_LIST:
    kap = build_kappa_map(nu_bins, n_true["HI"], n_true["HeI"], n_true["HeII"]) * DX
    tau_bins = np.array(kap.mean(axis=(0, 1, 2)) * Nx)          # mean LOS optical depth per bin
    tau_med = float(np.median(tau_bins))
    # signal: how much structure imprints on J vs a uniform-abundance box
    J_t = forward(n_true["HI"], n_true["HeI"], n_true["HeII"], DX, NR_SW)
    J_u = forward(ref["HI"], ref["HeI"], ref["HeII"], DX, NR_SW)
    dJ_rel = float(jnp.linalg.norm(((J_t - J_u) * mask).ravel()) / (jnp.linalg.norm((J_u * mask).ravel()) + EPS))
    # recoverable fraction from sensitivity
    S = sensitivity(DX); Si = np.array(S[1:-1, 1:-1, 1:-1]).ravel()
    order = np.argsort(Si)[::-1]; cs = np.cumsum(Si[order]); cs /= cs[-1] + EPS
    thr = Si[order[int(np.searchsorted(cs, F_CUM))]]
    frac = float((Si >= thr).mean())
    dead = float((Si < 0.01 * Si.max()).mean())
    print(f"{DX:10.0e} {tau_med:12.2f} {frac:11.3f} {dead:10.3f} {dJ_rel:9.3f}")
    rows.append((DX, tau_med, frac, dead, dJ_rel))

rows = np.array(rows)
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.semilogx(rows[:, 1], 100 * rows[:, 2], "o-", color="tab:blue", label="recoverable fraction [%]")
ax1.set_xlabel("median line-of-sight $\\tau_{box}$"); ax1.set_ylabel("recoverable fraction [%]", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue"); ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.semilogx(rows[:, 1], rows[:, 4], "s--", color="tab:red", label="signal $dJ_{rel}$")
ax2.set_ylabel("absorption signal $dJ_{rel}$", color="tab:red"); ax2.tick_params(axis="y", labelcolor="tab:red")
ax1.invert_xaxis()  # decreasing opacity -> rightward
plt.title("Opacity trade-off: recoverable fraction vs absorption signal")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "opacity_sweep.png"), dpi=180, bbox_inches="tight")
plt.close()
print("Saved opacity_sweep.png")
