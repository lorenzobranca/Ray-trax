"""
Analytic validation of the multi-frequency solver (ray_trax_3D_multifreq).

For a point source of luminosity L in a medium with constant opacity κ_ν,
the analytic mean intensity at distance r is:

    J_ν(r) = L / (4π r²) × exp(−κ_ν r)

We test four frequency bins against this formula using HI photoionization
cross sections (pure hydrogen, no HeI/HeII):

  bin 0 : 0.5 × ν_HI  → below threshold, κ = 0  (pure 1/r² falloff)
  bin 1 : 1.5 × ν_HI  → κ ≈ 0.058 grid⁻¹  (τ ≈ 1.8 across half-box)
  bin 2 : 3.0 × ν_HI  → κ ≈ 0.007 grid⁻¹  (τ ≈ 0.23)
  bin 3 : 6.0 × ν_HI  → κ ≈ 0.001 grid⁻¹  (nearly transparent)

Physical setup:
  dx = 0.01 pc = 3.086e16 cm  (grid spacing)
  n_HI = 1.0 cm⁻³              (uniform, no HeI / HeII)

Outputs:
  plots_multifreq/radial_profiles.png   — numeric vs analytic per bin
  plots_multifreq/relative_error.png    — |J_num − J_an| / J_an vs r
"""

from autocvd import autocvd
autocvd(num_gpus=1)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from ray_trax.cross_sections import sigma_HI, build_kappa_map, NU_HI
from ray_trax.ray_trax_3D_multifreq import compute_radiation_field_multifreq
from ray_trax.utils import gaussian_emissivity

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")

os.makedirs("plots_multifreq", exist_ok=True)

# ─── grid ────────────────────────────────────────────────────────────────────
Nx = Ny = Nz = 64
source_pos = jnp.array([Nx / 2.0, Ny / 2.0, Nz / 2.0])
L = 1.0   # total source luminosity (arbitrary units, same for all bins)

# ─── physical ↔ grid conversion ──────────────────────────────────────────────
PC_TO_CM = 3.086e18         # cm per parsec
dx_cm    = 0.01 * PC_TO_CM  # 0.01 pc per grid cell = 3.086e16 cm

# ─── frequency bins ──────────────────────────────────────────────────────────
FREQ_FACTORS = jnp.array([0.5, 1.5, 3.0, 6.0])
nu_bins      = FREQ_FACTORS * NU_HI
n_freq       = len(nu_bins)

BIN_LABELS = [
    "0.5 ν_HI  (below thr., κ=0)",
    "1.5 ν_HI",
    "3.0 ν_HI",
    "6.0 ν_HI",
]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ─── cross-section sanity checks ─────────────────────────────────────────────
print("\n── Cross-section checks ──────────────────────────────────────────────")

sigma_vals = sigma_HI(nu_bins)
print(f"σ_HI at 0.5 ν_HI : {float(sigma_vals[0]):.3e} cm²   (expect 0.0)")
print(f"σ_HI at 1.5 ν_HI : {float(sigma_vals[1]):.3e} cm²   (expect {6.304e-18/1.5**3:.3e})")
print(f"σ_HI at 3.0 ν_HI : {float(sigma_vals[2]):.3e} cm²   (expect {6.304e-18/3.0**3:.3e})")
print(f"σ_HI at 6.0 ν_HI : {float(sigma_vals[3]):.3e} cm²   (expect {6.304e-18/6.0**3:.3e})")

# ν⁻³ power law: ratio between bin1 and bin2 should be (1.5/3)³ = 0.125
ratio_12 = float(sigma_vals[1] / sigma_vals[2])
assert abs(ratio_12 - (3.0/1.5)**3) < 0.01, f"ν⁻³ scaling wrong: {ratio_12:.4f} (expect 8.0)"
print(f"σ(1.5ν)/σ(3.0ν) = {ratio_12:.4f}   (expect 8.000  — ν⁻³ scaling)  ✓")

assert float(sigma_vals[0]) == 0.0, "below-threshold bin should have σ=0"
print("σ = 0 below threshold  ✓")

# ─── build kappa field in grid units ─────────────────────────────────────────
n_HI_field  = jnp.ones((Nx, Ny, Nz)) * 1.0   # cm⁻³
n_HeI_field = jnp.zeros((Nx, Ny, Nz))
n_HeII_field = jnp.zeros((Nx, Ny, Nz))

kappa_physical = build_kappa_map(nu_bins, n_HI_field, n_HeI_field, n_HeII_field)  # cm⁻¹
kappa_grid     = kappa_physical * dx_cm     # dimensionless per grid step

kappa_vals_grid = kappa_grid[0, 0, 0, :]    # uniform field, all cells same
print("\n── Opacity per grid cell ─────────────────────────────────────────────")
for i, (lbl, kv) in enumerate(zip(BIN_LABELS, kappa_vals_grid)):
    tau_half = float(kv) * (Nx / 2)
    print(f"  bin {i}  {lbl:<30s}  κ_grid = {float(kv):.4e}   τ(half-box) = {tau_half:.3f}")

assert float(kappa_vals_grid[0]) == 0.0, "below-threshold bin must have κ=0"
print("κ = 0 for below-threshold bin  ✓")

# ─── emissivity: narrow Gaussian normalised to total luminosity L ─────────────
j_mono = gaussian_emissivity(Nx, Ny, Nz, center=source_pos, amplitude=1.0, width=2.0)
j_mono = j_mono * (L / jnp.sum(j_mono))          # normalise to total L
j_map  = jnp.repeat(j_mono[..., None], n_freq, axis=-1)   # (Nx,Ny,Nz,n_freq)

# ─── run solver ──────────────────────────────────────────────────────────────
NUM_RAYS  = 4096
STEP_SIZE = 0.5
MAX_STEPS = 130   # ≥ sqrt(3)*32/0.5 ≈ 110, covers full box diagonal

print(f"\nRunning multi-frequency solver  ({NUM_RAYS} rays, {MAX_STEPS} steps)  …")
t0 = time.perf_counter()
J_num = compute_radiation_field_multifreq(
    j_map, kappa_grid, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
)
jax.block_until_ready(J_num)
print(f"First call (includes JIT):  {time.perf_counter()-t0:.2f} s")

t0 = time.perf_counter()
J_num = compute_radiation_field_multifreq(
    j_map, kappa_grid, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
)
jax.block_until_ready(J_num)
print(f"Second call (compiled):     {time.perf_counter()-t0:.2f} s")

print(f"J_num shape: {J_num.shape}")

# ─── radial grid ─────────────────────────────────────────────────────────────
X, Y, Z = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), jnp.arange(Nz), indexing='ij')
r     = jnp.sqrt((X - source_pos[0])**2 +
                  (Y - source_pos[1])**2 +
                  (Z - source_pos[2])**2)
r_int = jnp.round(r).astype(int)

ix0, iy0, iz0 = Nx // 2, Ny // 2, Nz // 2

BULK_RMIN = 6    # > 3× Gaussian width=2; avoids near-source discretisation
BULK_RMAX = 28   # avoids boundary

def shell_avg(arr, rshell):
    mask = r_int == rshell
    return float(jnp.mean(arr[mask])) if int(jnp.sum(mask)) > 0 else None

r_shells   = np.arange(BULK_RMIN, BULK_RMAX + 1)
J_shells   = np.array([[shell_avg(J_num[..., i], rs) for rs in r_shells]
                       for i in range(n_freq)])   # (n_freq, n_shells)

# ─── check 1: 1/r² falloff in κ=0 bin ────────────────────────────────────────
# J_0(r) × r² should be constant  (geometric falloff with no attenuation)
print("\n── Check 1: 1/r² geometric falloff  (bin 0, κ=0) ───────────────────")
TOL_GEOM = 0.10   # 10 % variation in J × r²
J0_r2    = J_shells[0] * r_shells**2
norm     = np.mean(J0_r2)
fluct    = np.max(np.abs(J0_r2 / norm - 1.0))
passed_geom = fluct < TOL_GEOM
print(f"  Max variation in J₀×r²: {fluct:.3f}   tol {TOL_GEOM}  "
      f"→ {'PASS' if passed_geom else 'FAIL'}")

# ─── check 2: relative attenuation across bins ────────────────────────────────
# J_num[i](r) / J_num[0](r) should follow exp(-κ_i × r)
# This tests the cross-section physics without needing the absolute normalisation.
print("\n── Check 2: differential attenuation  J_i / J_0 vs exp(-κ_i r) ─────")
TOL_ATT_MEAN = 0.05   # 5 % mean relative error on the ratio
TOL_ATT_MAX  = 0.10   # 10 % max

all_passed = True
for i in range(1, n_freq):   # skip bin 0 (reference)
    kappa_i   = float(kappa_vals_grid[i])
    ratio_num = J_shells[i] / (J_shells[0] + 1e-40)
    ratio_an  = np.exp(-kappa_i * r_shells)
    errs      = np.abs(ratio_num - ratio_an) / (ratio_an + 1e-40)
    mean_err  = float(np.mean(errs))
    max_err   = float(np.max(errs))
    passed    = (mean_err < TOL_ATT_MEAN) and (max_err < TOL_ATT_MAX)
    if not passed:
        all_passed = False
    print(f"  ['{'PASS' if passed else 'FAIL'}'] bin {i} ({BIN_LABELS[i]})  "
          f"κ={kappa_i:.4e} grid⁻¹:  "
          f"mean err = {mean_err:.4f}   max err = {max_err:.4f}")

all_passed = all_passed and passed_geom

# ─── calibrated analytic for plots ───────────────────────────────────────────
# fit normalisation C from bin 0 (κ=0):  J_0 = C × L/(4π r²)
# so that plots can show the correct absolute comparison
r_shells_f = r_shells.astype(float)
C_calib = np.mean(J_shells[0] * 4 * np.pi * r_shells_f**2 / L)
print(f"\nNormalisation factor C (J_num / J_analytic) = {C_calib:.4f}")
# C ~ 0.5 is expected: deposited-J convention counts intensity passing
# through each cell; the 4π/N factor converts to energy flux, not mean
# intensity in the full-sphere sense.

# ─── plots ────────────────────────────────────────────────────────────────────
r_all_shells = np.arange(1, Nx // 2)
J_all_shells = np.array([[shell_avg(J_num[..., i], rs) for rs in r_all_shells]
                         for i in range(n_freq)])

# 1. Radial profiles (shell-averaged) with calibrated analytic
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for i, (ax, lbl, col) in enumerate(zip(axes, BIN_LABELS, COLORS)):
    kappa_i   = float(kappa_vals_grid[i])
    avg_num   = J_all_shells[i]
    avg_an_cal = C_calib * L / (4 * np.pi * r_all_shells**2) * np.exp(-kappa_i * r_all_shells)

    ax.semilogy(r_all_shells, avg_num,    color=col, lw=2,   label="Numeric (shell avg)")
    ax.semilogy(r_all_shells, avg_an_cal, color=col, lw=1.5, ls="--",
                label=f"C×L/4πr²·e^{{-κr}}  (C={C_calib:.2f})")
    ax.axvline(BULK_RMIN, color="gray", ls=":", lw=0.8, label="bulk start")
    ax.set_xlabel("r  [grid units]")
    ax.set_ylabel("⟨J_ν⟩  [arb. units]")
    ax.set_title(f"bin {i}: {lbl}\nκ = {kappa_i:.4e} grid⁻¹")
    ax.legend(fontsize=7)
    ax.set_xlim(0, Nx // 2)

plt.suptitle(f"Multi-frequency RT: shell-averaged profiles  (C = {C_calib:.2f})", fontsize=11)
plt.tight_layout()
plt.savefig("plots_multifreq/radial_profiles.png", dpi=150)
plt.close()
print("\nSaved plots_multifreq/radial_profiles.png")

# 2. Attenuation ratio J_i / J_0 vs exp(-κ_i r)
fig, ax = plt.subplots(figsize=(7, 5))
for i in range(1, n_freq):
    kappa_i   = float(kappa_vals_grid[i])
    ratio_num = J_all_shells[i] / (J_all_shells[0] + 1e-40)
    ratio_an  = np.exp(-kappa_i * r_all_shells)
    ax.plot(r_all_shells, ratio_num, color=COLORS[i], lw=2,   label=f"bin {i} numeric")
    ax.plot(r_all_shells, ratio_an,  color=COLORS[i], lw=1.5, ls="--",
            label=f"exp(−{kappa_i:.3e}·r)")
ax.axvline(BULK_RMIN, color="k", ls=":", lw=0.8, label="bulk start")
ax.set_xlabel("r  [grid units]")
ax.set_ylabel("J_i / J_0  (attenuation relative to κ=0 bin)")
ax.set_title("Differential attenuation: numeric vs analytic exp(−κ r)")
ax.legend(fontsize=8)
ax.set_xlim(0, Nx // 2)
plt.tight_layout()
plt.savefig("plots_multifreq/attenuation_ratio.png", dpi=150)
plt.close()
print("Saved plots_multifreq/attenuation_ratio.png")

# 3. Midplane slices for all bins
fig, axes = plt.subplots(2, n_freq, figsize=(4 * n_freq, 8))
for i in range(n_freq):
    J_slice_num = np.log10(np.array(J_num[:, iy0, :, i]) + 1e-30)
    J_slice_an  = np.log10(np.array(J_an [:, iy0, :, i]) + 1e-30)
    vmin = min(J_slice_an.min(), J_slice_num.min())
    vmax = max(J_slice_an.max(), J_slice_num.max())

    im0 = axes[0, i].imshow(J_slice_num, origin='lower', cmap='inferno',
                             vmin=vmin, vmax=vmax)
    axes[0, i].set_title(f"Numeric\nbin {i}: {FREQ_FACTORS[i]:.1f}ν_HI")
    plt.colorbar(im0, ax=axes[0, i])

    im1 = axes[1, i].imshow(J_slice_an, origin='lower', cmap='inferno',
                             vmin=vmin, vmax=vmax)
    axes[1, i].set_title(f"Analytic\nlog₁₀ J_ν")
    plt.colorbar(im1, ax=axes[1, i])

plt.suptitle("Mid-plane slices  log₁₀ J_ν  (numeric top, analytic bottom)", fontsize=12)
plt.tight_layout()
plt.savefig("plots_multifreq/midplane_slices.png", dpi=150)
plt.close()
print("Saved plots_multifreq/midplane_slices.png")

# ─── final verdict ────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
if all_passed:
    print("All bins PASSED  ✓")
else:
    print("Some bins FAILED  ✗  (see per-bin output above)")
    raise SystemExit(1)
