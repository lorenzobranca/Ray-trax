"""
degeneracy_sweep.py — identifiability / degeneracy study for field-level n_HI inference.

NOT an inversion. This forward-models the SAME physical setup as inference_field.py
(64^3 grid, 8 fixed sources, 6 freq bins, trilinear kappa) under many parameter
configurations, and measures how well the forward map J(n_HI) *separates* distinct
density fields. The goal is to find a regime where different n_HI give measurably
different J, so the later inverse problem is not hopelessly degenerate.

Degeneracy knob (per-step optical depth):
    d_tau = n_HI * sigma_HI(nu) * DX_PHYS * step_size
  tau << 1  -> rays unattenuated      -> J insensitive to n_HI  -> degenerate
  tau ~ O(1)-> J most sensitive to n_HI                          -> identifiable
  tau >> 1  -> saturated, deep shadows -> J insensitive there    -> degenerate

Diagnostics per config (all on the interior, boundary masked):
  1. Separation ratio across K reseeded lognormal fields:
        dJ_rel  = ||J_a - J_b|| / mean(||J||)        (absolute distinguishability)
        ratio   = dJ_rel / dn_rel                    (gain; >1 = J amplifies differences)
  2. Per-voxel sensitivity map  S(v) = |d(sum_masked J)/d n_HI(v)|  (one reverse-mode VJP).
     Low S => voxel lives in the forward-map nullspace (shadow) => unrecoverable.
  3. Optical-depth diagnostics: per-nu mean line-of-sight tau across the box.

Usage:
    python degeneracy_sweep.py --mode dx                      # sweep DX_PHYS at base field
    python degeneracy_sweep.py --mode field --dx_phys 3e16    # sweep sigma_g, length_scale
    python degeneracy_sweep.py --mode quick                   # fast smoke test
"""
from autocvd import autocvd
autocvd(num_gpus=1)

import os, time, json, argparse, itertools
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jax import random
from functools import partial

from ray_trax.ray_trax_3D_multifreq import compute_radiation_field_multifreq
from ray_trax.cross_sections import build_kappa_map, NU_HI

# ─── fixed physical setup (mirrors inference_field.py) ────────────────────────
GRID        = (64, 64, 64)
N_FREQ      = 6
N_SOURCES   = 8
STEP_SIZE   = 0.5
MAX_STEPS   = 256          # ~> box diagonal (111) / step_size
EPS         = 1e-10
SAVE_DIR    = "plots_degeneracy"
Nx, Ny, Nz = GRID
os.makedirs(SAVE_DIR, exist_ok=True)

# base field statistics (used as the fixed point when sweeping DX_PHYS)
BASE_SIGMA_G = 1.2
BASE_LSCALE  = 0.1

nu_bins = jnp.array(np.geomspace(NU_HI, 5.0 * NU_HI, N_FREQ), dtype=jnp.float32)
SED = (nu_bins / NU_HI) ** (-1.5)
SED = SED / (SED.sum() + EPS)

# ─── sources: identical to inference_field.py so the study is representative ──
_key = random.PRNGKey(42)
_, _srckey = random.split(_key)
source_positions = ((random.uniform(_srckey, (N_SOURCES, 3)) * 0.6 + 0.2)
                    * jnp.array([Nx, Ny, Nz], dtype=jnp.float32)).astype(jnp.float32)

_x = jnp.arange(Nx, dtype=jnp.float32)[:, None, None]
_y = jnp.arange(Ny, dtype=jnp.float32)[None, :, None]
_z = jnp.arange(Nz, dtype=jnp.float32)[None, None, :]

_j_list = []
for i in range(N_SOURCES):
    cx, cy, cz = (float(source_positions[i, 0]),
                  float(source_positions[i, 1]),
                  float(source_positions[i, 2]))
    e = jnp.exp(-0.5 * ((_x - cx) ** 2 + (_y - cy) ** 2 + (_z - cz) ** 2) / 2.0 ** 2)
    e = e / (e.sum() + EPS) * (0.5 / N_SOURCES)
    _j_list.append(e[..., None] * SED[None, None, None, :])
j_maps = jnp.stack(_j_list)               # (N_SOURCES, Nx, Ny, Nz, N_FREQ)

_zeros = jnp.zeros(GRID, dtype=jnp.float32)

# interior mask (drop 1-voxel boundary shell where ray tracing is least accurate)
mask = jnp.ones((*GRID, N_FREQ), dtype=jnp.float32)
mask = mask.at[:1].set(0);       mask = mask.at[-1:].set(0)
mask = mask.at[:, :1].set(0);    mask = mask.at[:, -1:].set(0)
mask = mask.at[:, :, :1].set(0); mask = mask.at[:, :, -1:].set(0)


# ─── field generator (matches inference_field.make_lognormal_field) ──────────
def make_lognormal_field(key, sigma_g, length_scale):
    kx = jnp.fft.fftfreq(Nx) / length_scale
    ky = jnp.fft.fftfreq(Ny) / length_scale
    kz = jnp.fft.fftfreq(Nz) / length_scale
    kx3, ky3, kz3 = jnp.meshgrid(kx, ky, kz, indexing='ij')
    k   = jnp.sqrt(kx3 ** 2 + ky3 ** 2 + kz3 ** 2)
    P_k = jnp.exp(-0.5 * (jnp.log(jnp.clip(k, 1e-6)) / 0.5) ** 2)
    P_k = P_k.at[0, 0, 0].set(0.0)
    phases = jnp.exp(2j * jnp.pi * random.uniform(key, GRID))
    g = jnp.fft.ifftn(jnp.sqrt(P_k) * phases).real
    g = sigma_g * (g - g.mean()) / (g.std() + EPS)
    field = jnp.exp(g)
    return (field / (field.mean() + EPS)).astype(jnp.float32)   # mean-normalised to 1


# ─── forward model (compiled once; reused across the whole sweep) ────────────
@partial(jax.jit, static_argnames=["num_rays", "ray_batch_size"])
def forward(n_HI, dx_phys, num_rays, ray_batch_size):
    kappa = build_kappa_map(nu_bins, n_HI, _zeros, _zeros) * dx_phys
    J = jnp.zeros((*GRID, N_FREQ), dtype=jnp.float32)
    for i in range(N_SOURCES):
        J = J + compute_radiation_field_multifreq(
            j_maps[i], kappa, source_positions[i],
            num_rays=num_rays, step_size=STEP_SIZE, max_steps=MAX_STEPS,
            ray_batch_size=ray_batch_size, kappa_interp="trilinear",
        )
    return J


@partial(jax.jit, static_argnames=["num_rays", "ray_batch_size"])
def sensitivity_map(n_HI, dx_phys, num_rays, ray_batch_size):
    """S(v) = d( sum_masked J ) / d n_HI(v)  — one reverse-mode VJP."""
    def scalar(n):
        return jnp.sum(mask * forward(n, dx_phys, num_rays, ray_batch_size))
    return jax.grad(scalar)(n_HI)


# ─── per-config diagnostics ──────────────────────────────────────────────────
def masked_norm(a):
    return jnp.sqrt(jnp.sum((mask * a) ** 2))


def run_config(dx_phys, sigma_g, length_scale, k_fields, num_rays, ray_batch_size,
               seed0=1000, tag=""):
    t0 = time.time()

    # K reseeded lognormal fields with this config's statistics
    fields = [make_lognormal_field(random.PRNGKey(seed0 + s), sigma_g, length_scale)
              for s in range(k_fields)]
    Js = [forward(f, dx_phys, num_rays, ray_batch_size) for f in fields]
    jax.block_until_ready(Js[-1])

    # 1. separation across all pairs
    ratios, dJ_rels, dn_rels = [], [], []
    for a, b in itertools.combinations(range(k_fields), 2):
        nJ = 0.5 * (masked_norm(Js[a]) + masked_norm(Js[b]))
        nn = 0.5 * (jnp.linalg.norm(fields[a]) + jnp.linalg.norm(fields[b]))
        dJ = float(masked_norm(Js[a] - Js[b]) / (nJ + EPS))
        dn = float(jnp.linalg.norm(fields[a] - fields[b]) / (nn + EPS))
        dJ_rels.append(dJ); dn_rels.append(dn); ratios.append(dJ / (dn + EPS))

    # 2. per-voxel sensitivity (on field 0)
    S = jnp.abs(sensitivity_map(fields[0], dx_phys, num_rays, ray_batch_size))
    S_int = S * mask[..., 0]
    Smax = float(S_int.max()) + EPS
    interior = float(mask[..., 0].sum())
    dead_frac = float(jnp.sum((S_int < 0.01 * Smax) * mask[..., 0]) / interior)
    S_med = float(jnp.median(S_int[mask[..., 0] > 0]))
    dyn_range = Smax / (S_med + EPS)

    # 3. optical depth: mean line-of-sight tau across the box, per nu bin
    kappa = build_kappa_map(nu_bins, fields[0], _zeros, _zeros) * dx_phys  # (...,N_FREQ)
    tau_box = np.array(jnp.mean(kappa, axis=(0, 1, 2)) * Nx)               # (N_FREQ,)

    res = dict(
        tag=tag, dx_phys=float(dx_phys), sigma_g=float(sigma_g),
        length_scale=float(length_scale), num_rays=int(num_rays), k_fields=int(k_fields),
        dJ_rel_mean=float(np.mean(dJ_rels)), dJ_rel_min=float(np.min(dJ_rels)),
        dn_rel_mean=float(np.mean(dn_rels)),
        ratio_mean=float(np.mean(ratios)), ratio_min=float(np.min(ratios)),
        dead_frac=dead_frac, sens_dyn_range=float(dyn_range),
        tau_box_min=float(tau_box.min()), tau_box_max=float(tau_box.max()),
        tau_box_med=float(np.median(tau_box)),
        tau_box_per_bin=[float(t) for t in tau_box],
        runtime_s=round(time.time() - t0, 1),
    )

    # sensitivity slice plot for this config
    mid = Nz // 2
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    im = ax[0].imshow(np.array(fields[0][:, :, mid]), origin='lower', cmap='viridis')
    ax[0].set_title("n_HI (field 0)"); plt.colorbar(im, ax=ax[0])
    Sslice = np.log10(np.array(S_int[:, :, mid]) / Smax + 1e-6)
    im = ax[1].imshow(Sslice, origin='lower', cmap='magma', vmin=-6, vmax=0)
    ax[1].set_title("log10 sensitivity |dJ/dn_HI| (norm.)"); plt.colorbar(im, ax=ax[1])
    src = np.array(source_positions); near = np.abs(src[:, 2] - mid) < 4
    for a in ax:
        a.scatter(src[near, 1], src[near, 0], c='cyan', marker='+', s=60, linewidths=1.5)
    plt.suptitle(f"{tag}  |  dJ_rel={res['dJ_rel_mean']:.2e}  ratio={res['ratio_mean']:.2f}  "
                 f"dead={dead_frac:.2f}  tau_box=[{res['tau_box_min']:.2g},{res['tau_box_max']:.2g}]")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"sens_{tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[{tag}] dJ_rel={res['dJ_rel_mean']:.3e} (min {res['dJ_rel_min']:.2e}) | "
          f"ratio={res['ratio_mean']:.3f} | dead_frac={dead_frac:.3f} | "
          f"tau_box=[{res['tau_box_min']:.3g}, {res['tau_box_max']:.3g}] | {res['runtime_s']}s")
    return res


def append_results(rows):
    path = os.path.join(SAVE_DIR, "results.jsonl")
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  appended {len(rows)} row(s) -> {path}")


# ─── sweep modes ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dx", "field", "quick", "one"], default="dx")
    ap.add_argument("--dx_phys", type=float, default=3e16,
                    help="fixed DX_PHYS for --mode field / value for --mode one")
    ap.add_argument("--sigma_g", type=float, default=BASE_SIGMA_G, help="for --mode one")
    ap.add_argument("--length_scale", type=float, default=BASE_LSCALE, help="for --mode one")
    ap.add_argument("--rays", type=int, default=3000)
    ap.add_argument("--kfields", type=int, default=4)
    ap.add_argument("--rbs", type=int, default=50, help="ray_batch_size")
    args = ap.parse_args()

    print(f"mode={args.mode} rays={args.rays} kfields={args.kfields} "
          f"grid={GRID} sources={N_SOURCES} freqs={N_FREQ}")
    rows = []

    if args.mode == "one":
        rows.append(run_config(args.dx_phys, args.sigma_g, args.length_scale,
                               k_fields=args.kfields, num_rays=args.rays,
                               ray_batch_size=args.rbs,
                               tag=f"sg{args.sigma_g}_ls{args.length_scale}_dx{args.dx_phys:.0e}"))

    elif args.mode == "quick":
        rows.append(run_config(5e15, BASE_SIGMA_G, BASE_LSCALE,
                               k_fields=2, num_rays=300, ray_batch_size=args.rbs,
                               tag="quick_dx5e15"))

    elif args.mode == "dx":
        dx_grid = [1e15, 3e15, 1e16, 3e16, 1e17]
        for dx in dx_grid:
            rows.append(run_config(dx, BASE_SIGMA_G, BASE_LSCALE,
                                   k_fields=args.kfields, num_rays=args.rays,
                                   ray_batch_size=args.rbs,
                                   tag=f"dx{dx:.0e}"))

    elif args.mode == "field":
        sigma_grid  = [0.8, 1.2, 1.8]
        lscale_grid = [0.05, 0.10, 0.20]
        for sg, ls in itertools.product(sigma_grid, lscale_grid):
            rows.append(run_config(args.dx_phys, sg, ls,
                                   k_fields=args.kfields, num_rays=args.rays,
                                   ray_batch_size=args.rbs,
                                   tag=f"sg{sg}_ls{ls}_dx{args.dx_phys:.0e}"))

    append_results(rows)
    print("\nSummary (this run):")
    for r in sorted(rows, key=lambda d: -d["dJ_rel_mean"]):
        print(f"  {r['tag']:>24s} | dJ_rel={r['dJ_rel_mean']:.3e} | "
              f"ratio={r['ratio_mean']:6.3f} | dead={r['dead_frac']:.3f} | "
              f"tau_box=[{r['tau_box_min']:.3g},{r['tau_box_max']:.3g}]")


if __name__ == "__main__":
    main()
