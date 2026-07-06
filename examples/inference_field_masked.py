"""
inference_field_masked.py — well-posed, restricted field-level inference of n_HI.

Motivation
----------
With point sources, the forward map J(n_HI) has a ~99% nullspace: flux (and hence
sensitivity |dJ/dn_HI|) falls off like 1/r^2 from each source, so only voxels in
the illuminated core carry a measurable absorption signal (see degeneracy_sweep.py:
sens_dyn_range ~ 1e4, dead_frac ~ 0.99). Inverting the *full* field is therefore
hopelessly degenerate.

This script makes the inverse problem well-posed by RESTRICTING the target to the
recoverable subdomain:

  1. Define a recoverability mask M_rec from the forward map's own sensitivity,
     evaluated at a UNIFORM reference field (truth-independent — depends only on
     source geometry, not on the answer).
  2. Optimize n_HI only on M_rec; freeze the rest at a prior n_bg.
  3. Report rel_err(n_HI) over M_rec (the well-posed claim) vs the complement.

Usage:
    python inference_field_masked.py                 # f_cum=0.90
    python inference_field_masked.py --f_cum 0.95
    python inference_field_masked.py --f_cum 0.80 --iters 400
"""
from autocvd import autocvd
autocvd(num_gpus=1)

import os, time, argparse
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
NUM_RAYS       = 4000
RAY_BATCH_SIZE = 50
STEP_SIZE      = 0.5
MAX_STEPS      = 320
LR             = 3e-3
EPS            = 1e-10
DX_PHYS        = 5e15          # cm per grid unit
N_BG           = 1.0          # prior value for frozen (non-recoverable) voxels
SAVE_DIR       = "plots_field_inference_masked"
os.makedirs(SAVE_DIR, exist_ok=True)
Nx, Ny, Nz = GRID

nu_bins = jnp.array(np.geomspace(NU_HI, 5.0 * NU_HI, N_FREQ), dtype=jnp.float32)


# ─── true n_HI: spatially correlated lognormal field ─────────────────────────
def make_lognormal_field(key, shape, sigma_g=1.2, length_scale=0.1):
    Nx, Ny, Nz = shape
    kx = jnp.fft.fftfreq(Nx) / length_scale
    ky = jnp.fft.fftfreq(Ny) / length_scale
    kz = jnp.fft.fftfreq(Nz) / length_scale
    kx3, ky3, kz3 = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k   = jnp.sqrt(kx3**2 + ky3**2 + kz3**2)
    P_k = jnp.exp(-0.5 * (jnp.log(jnp.clip(k, 1e-6)) / 0.5) ** 2)
    P_k = P_k.at[0, 0, 0].set(0.0)
    phases = jnp.exp(2j * jnp.pi * random.uniform(key, shape))
    g = jnp.fft.ifftn(jnp.sqrt(P_k) * phases).real
    g = sigma_g * (g - g.mean()) / (g.std() + EPS)
    field = jnp.exp(g)
    return field / (field.mean() + EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f_cum", type=float, default=0.90,
                    help="fraction of total sensitivity the recoverable mask must capture")
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()

    key       = random.PRNGKey(42)
    n_HI_true = make_lognormal_field(key, GRID).astype(jnp.float32)
    print(f"n_HI true: min={float(n_HI_true.min()):.3f}, "
          f"max={float(n_HI_true.max()):.3f}, mean={float(n_HI_true.mean()):.3f}")

    # ─── sources: 8 random positions, total luminosity = 0.5 ─────────────────
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
        cx, cy, cz = (float(source_positions[i, 0]),
                      float(source_positions[i, 1]),
                      float(source_positions[i, 2]))
        e = jnp.exp(-0.5 * ((x_ - cx)**2 + (y_ - cy)**2 + (z_ - cz)**2) / 2.0**2)
        e = e / (e.sum() + EPS) * (0.5 / N_SOURCES)
        j_maps_list.append(e[..., None] * SED[None, None, None, :])
    j_maps = jnp.stack(j_maps_list)   # (N_SOURCES, Nx, Ny, Nz, N_FREQ)

    _zeros = jnp.zeros(GRID, dtype=jnp.float32)

    def forward(n_HI):
        kappa = build_kappa_map(nu_bins, n_HI, _zeros, _zeros) * DX_PHYS
        J = jnp.zeros((*GRID, N_FREQ), dtype=jnp.float32)
        for i in range(N_SOURCES):
            J = J + compute_radiation_field_multifreq(
                j_maps[i], kappa, source_positions[i],
                num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
                ray_batch_size=RAY_BATCH_SIZE, kappa_interp="trilinear",
            )
        return J

    # boundary mask on J (ray-tracing least accurate at the walls)
    Jmask = jnp.ones((*GRID, N_FREQ), dtype=jnp.float32)
    Jmask = Jmask.at[:1].set(0);       Jmask = Jmask.at[-1:].set(0)
    Jmask = Jmask.at[:, :1].set(0);    Jmask = Jmask.at[:, -1:].set(0)
    Jmask = Jmask.at[:, :, :1].set(0); Jmask = Jmask.at[:, :, -1:].set(0)

    # ─── reference J_obs from the true field ─────────────────────────────────
    print("\nGenerating J_obs from true n_HI...")
    t0    = time.time()
    J_obs = forward(n_HI_true)
    jax.block_until_ready(J_obs)
    print(f"  shape {J_obs.shape}, computed in {time.time()-t0:.2f}s")

    # ─── recoverability mask from forward-map sensitivity ────────────────────
    # S(v) = |d( sum_masked J )/d n_HI(v)|, evaluated at a UNIFORM reference
    # field so the mask depends only on source geometry, not on the truth.
    @jax.jit
    def sensitivity(n_ref):
        return jax.grad(lambda n: jnp.sum(Jmask * forward(n)))(n_ref)

    print("\nComputing sensitivity map at uniform reference field...")
    t0 = time.time()
    n_ref = jnp.full(GRID, N_BG, dtype=jnp.float32)
    S = jnp.abs(sensitivity(n_ref))                      # (Nx,Ny,Nz)
    S = S * Jmask[..., 0]                                 # drop boundary shell
    jax.block_until_ready(S)
    print(f"  done in {time.time()-t0:.2f}s | Smax={float(S.max()):.3e} "
          f"dyn_range(max/med)={float(S.max())/(float(jnp.median(S[S>0]))+EPS):.1f}")

    # smallest set of voxels capturing f_cum of total sensitivity
    S_flat = np.array(S).ravel()
    order  = np.argsort(S_flat)[::-1]
    csum   = np.cumsum(S_flat[order])
    csum  /= csum[-1] + EPS
    n_keep = int(np.searchsorted(csum, args.f_cum) + 1)
    thr    = float(S_flat[order[n_keep - 1]])
    M_rec  = (S >= thr).astype(jnp.float32)              # (Nx,Ny,Nz)
    frac   = float(M_rec.sum()) / (Nx * Ny * Nz)
    print(f"\nRecoverable mask: f_cum={args.f_cum} -> threshold={thr:.3e}")
    print(f"  voxels kept: {int(M_rec.sum())} / {Nx*Ny*Nz}  ({100*frac:.2f}% of box)")

    # ─── restricted parametrization ──────────────────────────────────────────
    # n_HI = M_rec * exp(log_n) + (1 - M_rec) * N_BG
    # gradient w.r.t. log_n is automatically zero off M_rec.
    def assemble(log_n):
        return M_rec * jnp.exp(log_n) + (1.0 - M_rec) * N_BG

    def loss_fn(log_n):
        J_pred = forward(assemble(log_n))
        diff   = J_pred - J_obs
        return jnp.mean(Jmask * (diff / (jnp.abs(J_obs) + EPS)) ** 2)

    optimizer = optax.adam(LR)
    log_n     = jnp.zeros(GRID, dtype=jnp.float32)        # init n_HI = 1 everywhere
    opt_state = optimizer.init(log_n)

    @jax.jit
    def step_fn(log_n, opt_state):
        loss, grads        = jax.value_and_grad(loss_fn)(log_n)
        updates, new_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(log_n, updates), new_state, loss

    # rel-err helpers restricted to a mask
    def rel_err_on(n_est, m):
        d = (n_est - n_HI_true) * m
        return float(jnp.linalg.norm(d) / (jnp.linalg.norm(n_HI_true * m) + EPS))

    M_comp = (1.0 - M_rec) * Jmask[..., 0]               # complement (interior only)

    # ─── optimize ────────────────────────────────────────────────────────────
    print(f"\nRunning {args.iters} Adam steps (lr={LR}) on {int(M_rec.sum())} voxels...")
    losses, errs_rec = [], []
    t0 = time.time()
    for i in range(args.iters):
        log_n, opt_state, loss_val = step_fn(log_n, opt_state)
        losses.append(float(loss_val))
        if i % 25 == 0 or i == args.iters - 1:
            n_cur = assemble(log_n)
            e_rec = rel_err_on(n_cur, M_rec)
            errs_rec.append((i, e_rec))
            print(f"  iter {i:4d} | loss={float(loss_val):.3e} | "
                  f"rel_err(M_rec)={e_rec:.3e}")
    print(f"Done in {time.time()-t0:.1f}s")

    # ─── final evaluation ────────────────────────────────────────────────────
    n_HI_rec = assemble(log_n)
    err_rec  = rel_err_on(n_HI_rec, M_rec)
    err_comp = rel_err_on(n_HI_rec, M_comp)
    err_full = rel_err_on(n_HI_rec, Jmask[..., 0])
    print(f"\nFinal rel L2(n_HI)  on M_rec      : {err_rec:.3e}   <-- well-posed target")
    print(f"Final rel L2(n_HI)  on complement : {err_comp:.3e}   (frozen at prior)")
    print(f"Final rel L2(n_HI)  on full interior: {err_full:.3e}")

    # ─── plots ────────────────────────────────────────────────────────────────
    mid = Nz // 2
    src = np.array(source_positions); near = np.abs(src[:, 2] - mid) < 4

    # (1) mask + sensitivity
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    im = ax[0].imshow(np.array(n_HI_true[:, :, mid]), origin='lower', cmap='viridis')
    ax[0].set_title("True $n_{HI}$"); plt.colorbar(im, ax=ax[0])
    Sslice = np.log10(np.array(S[:, :, mid]) / (float(S.max()) + EPS) + 1e-6)
    im = ax[1].imshow(Sslice, origin='lower', cmap='magma', vmin=-6, vmax=0)
    ax[1].set_title("log10 sensitivity (norm.)"); plt.colorbar(im, ax=ax[1])
    im = ax[2].imshow(np.array(M_rec[:, :, mid]), origin='lower', cmap='Greys_r')
    ax[2].set_title(f"recoverable mask ({100*frac:.1f}%)"); plt.colorbar(im, ax=ax[2])
    for a in ax:
        a.scatter(src[near, 1], src[near, 0], c='cyan', marker='+', s=60, linewidths=1.5)
    plt.suptitle(f"Recoverability: f_cum={args.f_cum}, {int(M_rec.sum())} voxels")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "recoverable_mask.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # (2) reconstruction on the mask
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    true_m = np.array((n_HI_true * M_rec)[:, :, mid])
    rec_m  = np.array((n_HI_rec  * M_rec)[:, :, mid])
    vmin, vmax = float(n_HI_true[:, :, mid].min()), float(n_HI_true[:, :, mid].max())
    im = ax[0].imshow(np.ma.masked_where(np.array(M_rec[:, :, mid]) == 0, true_m),
                      origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0].set_title("True $n_{HI}$ (on mask)"); plt.colorbar(im, ax=ax[0])
    im = ax[1].imshow(np.ma.masked_where(np.array(M_rec[:, :, mid]) == 0, rec_m),
                      origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1].set_title("Recovered $n_{HI}$ (on mask)"); plt.colorbar(im, ax=ax[1])
    err = np.abs(rec_m - true_m)
    im = ax[2].imshow(np.ma.masked_where(np.array(M_rec[:, :, mid]) == 0, err),
                      origin='lower', cmap='magma')
    ax[2].set_title("|Error| (on mask)"); plt.colorbar(im, ax=ax[2])
    for a in ax:
        a.scatter(src[near, 1], src[near, 0], c='white', marker='+', s=60, linewidths=1.5)
    plt.suptitle(f"Restricted inference | rel L2 on M_rec = {err_rec:.2e} "
                 f"(complement {err_comp:.2e})")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "masked_reconstruction.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # (3) loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(losses)
    ax.set_xlabel("Adam iteration"); ax.set_ylabel("masked rel L2 loss")
    ax.set_title(f"Restricted inference ({int(M_rec.sum())} voxels, f_cum={args.f_cum})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved figures to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
