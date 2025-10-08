# inference.py — cleaned
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# ------------------------- config -------------------------
GRID = (64, 64, 64)          # Nx, Ny, Nz
KAPPA0 = 0.1
NUM_RAYS = 4000
STEP_SIZE = 0.5
MAX_STEPS = 400              # steady-only
LOSS_TYPE = "relative"       # "relative" or "mse"
EPS = 1e-10
A_TRUE = 2.0
SAVE_DIR = "plots_inference"
os.makedirs(SAVE_DIR, exist_ok=True)
# ------------------------- config -------------------------
FORCE_MODE = "tdep"  # set to "steady" or "tdep" to force, or None for auto

# ------------------------- solver selection -------------------------
if FORCE_MODE == "steady":
    from ray_trax.ray_trax_3D import compute_radiation_field_from_source as _steady
    _compute = ("steady", _steady)
elif FORCE_MODE == "tdep":
    from ray_trax.ray_trax_3D_tdep import compute_radiation_field_from_source_with_time_step as _tdep
    _compute = ("tdep", _tdep)
else:
    _compute = None
    try:
        from ray_trax.ray_trax_3D import compute_radiation_field_from_source as _steady
        _compute = ("steady", _steady)
    except Exception:
        from ray_trax.ray_trax_3D_tdep import compute_radiation_field_from_source_with_time_step as _tdep
        _compute = ("tdep", _tdep)

MODE, COMPUTE_FN = _compute
print(f"Using ray tracer mode: {MODE}")

# ------------------------- helpers -------------------------
def gaussian_emissivity(nx, ny, nz, center, width=2.0):
    x = jnp.arange(nx)[:, None, None]
    y = jnp.arange(ny)[None, :, None]
    z = jnp.arange(nz)[None, None, :]
    dx = x - center[0]; dy = y - center[1]; dz = z - center[2]
    return jnp.exp(-0.5 * (dx*dx + dy*dy + dz*dz) / (width**2))

def make_mask(shape, border=1):
    m = jnp.ones(shape)
    if border > 0:
        m = m.at[:border, :, :].set(0); m = m.at[-border:, :, :].set(0)
        m = m.at[:, :border, :].set(0); m = m.at[:, -border:, :].set(0)
        m = m.at[:, :, :border].set(0); m = m.at[:, :, -border:].set(0)
    return m

def forward_J(emissivity, *, kappa, source_pos):
    if MODE == "steady":
        return COMPUTE_FN(
            emissivity, kappa,
            source_pos=source_pos,
            num_rays=NUM_RAYS,
            step_size=STEP_SIZE,
            max_steps=MAX_STEPS,
        )
    else:
        nx, ny, nz = emissivity.shape
        
        # build a JAX scalar from the ints; use emissivity dtype
        box_diag = jnp.sqrt(jnp.array(nx*nx + ny*ny + nz*nz, dtype=emissivity.dtype))
        time_step = 1.2 * box_diag * jnp.asarray(STEP_SIZE, dtype=emissivity.dtype)
        
   
        return COMPUTE_FN(
            emissivity, kappa, source_pos,
            num_rays=NUM_RAYS,
            step_size=STEP_SIZE,
            time_step=time_step,
            radiation_velocity=1.0,
        )

# ------------------------- problem setup -------------------------
Nx, Ny, Nz = GRID
source_pos = jnp.array([Nx/2, Ny/2, Nz/2])
kappa = jnp.ones(GRID) * KAPPA0

e0 = gaussian_emissivity(Nx, Ny, Nz, center=source_pos, width=2.0)
e0 = e0 / (jnp.sum(e0) + 1e-12)   # sum(e0)=1 -> A equals total luminosity

# Build reference
t0 = time.time()
J_ref = forward_J(A_TRUE * e0, kappa=kappa, source_pos=source_pos)
jax.block_until_ready(J_ref)
print(f"Built reference field in {time.time()-t0:.2f}s")

# Mask and denominator
mask = make_mask(GRID, border=1).astype(J_ref.dtype)
mask_sum = jnp.sum(mask) + 1e-12

# Kernel for A=1 (also linearity check)
K = forward_J(e0, kappa=kappa, source_pos=source_pos)
K2 = forward_J(2*e0, kappa=kappa, source_pos=source_pos)
lin_err = float(jnp.linalg.norm(K2 - 2*K) / (jnp.linalg.norm(K) + 1e-12))
print("linearity err:", f"{lin_err:.2e}")

# Closed-form A* for MSE (useful baseline)
A_star_mse = float(jnp.sum(mask * K * J_ref) / (jnp.sum(mask * K * K) + 1e-12))
print("A* (closed form, MSE) =", f"{A_star_mse:.6f}")

# ------------------------- loss -------------------------
if LOSS_TYPE == "mse":
    def loss_from_amp(A):
        diff = forward_J(A * e0, kappa=kappa, source_pos=source_pos) - J_ref
        return jnp.sum(mask * diff * diff) / mask_sum
else:  # "relative"
    def loss_from_amp(A):
        J_pred = forward_J(A * e0, kappa=kappa, source_pos=source_pos)
        diff = J_pred - J_ref
        return jnp.sum(mask * (diff * diff) / (J_ref * J_ref + EPS)) / mask_sum


loss_jit = jax.jit(loss_from_amp)
grad_jit = jax.jit(jax.jacfwd(loss_from_amp))  # forward-mode grad (scalar A)

def loss_and_grad(A):
    return loss_jit(A), grad_jit(A)

# (optional) forward-mode "Hessian" for Newton:
hess_A = jax.jit(jax.jacfwd(jax.jacfwd(loss_from_amp)))


# ------------------------- optimize A (Newton-damped) -------------------------
A = jnp.array(0.5)   # poor initial guess on purpose
DAMP = 1e-6
MAX_STEP = 5.0
for it in range(26):
    Lval, gA = loss_and_grad(A)
    hA = hess_A(A)
    delta = gA / (hA + DAMP)
    delta = jnp.clip(delta, -MAX_STEP, MAX_STEP)
    A = jnp.clip(A - delta, 0.0)  # optional non-negativity
    if it % 5 == 0:
        print(f"iter {it:02d} | loss={float(Lval):.3e} | A={float(A):.6f} | grad={float(gA):.3e}")

print("\n--- Result ---")
print(f"True A     : {A_TRUE:.6f}")
print(f"Recovered A: {float(A):.6f}")

J_final = forward_J(A * e0, kappa=kappa, source_pos=source_pos)
rel_l2 = jnp.linalg.norm((J_final - J_ref).reshape(-1)) / (jnp.linalg.norm(J_ref.reshape(-1)) + EPS)
print(f"Relative L2(J): {float(rel_l2):.3e}")

# ------------------------- plotting: loss landscape -------------------------
def plot_loss_landscape(
    A_min=0.0, A_max=3.0, n=241,
    loss_type=LOSS_TYPE,
    savepath=os.path.join(SAVE_DIR, "loss_landscape_A.png"),
    mark_true=A_TRUE,
    check_linearity=True
):
    """
    Plot L(A) and dL/dA with clear, color-coded styling.
    Uses precomputed K = forward_J(e0) from the outer scope (for speed).
    """

    # --- optional linearity check (K is assumed precomputed globally) ---
    if check_linearity:
        K2 = forward_J(2 * e0, kappa=kappa, source_pos=source_pos)
        lin_err = float(jnp.linalg.norm(K2 - 2 * K) / (jnp.linalg.norm(K) + 1e-12))
        print(f"[loss landscape] linearity err: {lin_err:.2e}")

    # --- loss/grad definitions ---
    if loss_type == "mse":
        def L(A):
            diff = (A * K - J_ref)
            return jnp.sum(mask * diff * diff) / mask_sum
        def dL(A):
            return 2.0 * jnp.sum(mask * K * (A * K - J_ref)) / mask_sum
        A_star = float(jnp.sum(mask * K * J_ref) / (jnp.sum(mask * K * K) + 1e-12))
    else:
        w = mask / (J_ref * J_ref + EPS)
        def L(A):
            diff = (A * K - J_ref)
            return jnp.sum(w * diff * diff) / mask_sum
        def dL(A):
            return 2.0 * jnp.sum(w * K * (A * K - J_ref)) / mask_sum
        A_star = float(jnp.sum(w * K * J_ref) / (jnp.sum(w * K * K) + 1e-12))

    # --- sample grid and evaluate ---
    A_grid   = jnp.linspace(A_min, A_max, n)
    L_vals   = jax.vmap(L)(A_grid)
    dL_vals  = jax.vmap(dL)(A_grid)

    i_min     = int(jnp.argmin(L_vals))
    A_argmin  = float(A_grid[i_min])
    L_min     = float(L_vals[i_min])
    L_at_true = float(L(float(mark_true))) if mark_true is not None else None

    # --- formatting helpers ---
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    # Colorblind-friendly Tableau palette
    colors = {
        "loss":   "#1f77b4",  # blue
        "grad":   "#d62728",  # red
        "argmin": "#2ca02c",  # green
        "astar":  "#9467bd",  # purple
        "true":   "#ff7f0e",  # orange
    }

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))

    # L(A)
    ax1.plot(np.asarray(A_grid), np.asarray(L_vals),
             color=colors["loss"], lw=2.2, label=r"Loss $L(A)$")
    ax1.set_xlabel(r"Amplitude $A$")
    ax1.set_ylabel(r"Loss $L$", color=colors["loss"])
    ax1.tick_params(axis='y', colors=colors["loss"])
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.grid(True, which="major", alpha=0.35)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", alpha=0.15)
    ax1.margins(x=0)

    # Vertical markers (argmin, A*, A_true)
    ax1.axvline(A_argmin, color=colors["argmin"], linestyle="--", lw=1.8,
                label=fr"$\arg\min L \approx {A_argmin:.4f}$")
    ax1.axvline(A_star, color=colors["astar"], linestyle=":", lw=2.0,
                label=fr"$A^\star \approx {A_star:.4f}$")
    if mark_true is not None:
        ax1.axvline(float(mark_true), color=colors["true"], linestyle="-.", lw=1.6,
                    label=fr"$A_\mathrm{{true}} = {float(mark_true):.4f}$")
        if L_at_true is not None and np.isfinite(L_at_true):
            ax1.plot([float(mark_true)], [L_at_true], marker="o", ms=4,
                     color=colors["true"])

    # dL/dA on twin axis
    ax2 = ax1.twinx()
    ax2.plot(np.asarray(A_grid), np.asarray(dL_vals),
             color=colors["grad"], lw=1.9, alpha=0.95,
             label=r"$\mathrm{d}L/\mathrm{d}A$")
    ax2.set_ylabel(r"$\mathrm{d}L/\mathrm{d}A$", color=colors["grad"])
    ax2.tick_params(axis='y', colors=colors["grad"])
    ax2.axhline(0.0, lw=1.0, color=colors["grad"], alpha=0.35)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_alpha(0.9)

    # Nice title (optional): indicate loss type
    ax1.set_title(f"Loss landscape vs A  —  type: {loss_type.upper()}")

    plt.tight_layout()
    plt.savefig(savepath, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"[loss landscape] saved -> {savepath}")
    print(f"[loss landscape] argmin ~ {A_argmin:.6f}, loss ~ {L_min:.3e}")
    print(f"[loss landscape] A* (closed form) ~ {A_star:.6f}")


