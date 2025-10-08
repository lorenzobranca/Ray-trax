# Ray-trax 3D

**Time-dependent radiative transfer in 3D with JAX** — fast, differentiable ray marching for the monochromatic emission–absorption equation on regular grids. Built for research-grade workloads, GPU acceleration, and gradient-based workflows.

> 3D is the primary, stable target of this repository.

---

## Table of contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart (3D, time-dependent)](#quickstart-3d-time-dependent)
- [API at a glance](#api-at-a-glance)
- [Differentiability](#differentiability)
- [Testing](#testing)
- [Performance & scaling](#performance--scaling)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

Ray-trax 3D implements a **spatial marching** scheme for the time-dependent, monochromatic transport equation
\[
\frac{\partial I}{\partial t} + c\,\hat{\mathbf{n}}\!\cdot\!\nabla I = -\kappa(\mathbf{x})\,I + j(\mathbf{x}),
\]
with **3D** fields for opacity \(\kappa(\mathbf{x})\) and emissivity \(j(\mathbf{x})\). Each time step advances rays over a distance \(c\,\Delta t\) using semi-analytic attenuation, trilinear **interpolation/deposition** for smooth gradients, and near-uniform **Fibonacci** directions for robust angular coverage. Everything is **JIT-compiled** and **autodiff-friendly**.

---

## Features

- **3D-first, time-dependent** ray marching (single- and multi-source).
- **Vectorized** directional sampling via **Fibonacci/golden-angle** lattice.
- **Trilinear** interpolation and symmetric deposition (smooth w.r.t. voxel values).
- **Batch-first** design: directions, sources, and frequency bins as batch axes.
- **JIT + vmap** by default; optional **multi-device sharding** (rays per device).
- **End-to-end differentiability** (reverse-mode with static loops; forward-mode for dynamic horizons).
- Simple plotting helpers for slices and per-bin snapshots.

---

## Requirements

- Python **3.10+**
- `jax`, `jaxlib`
- `numpy`, `matplotlib`
- (Optional) GPU acceleration via CUDA/ROCm-enabled `jaxlib` (see JAX docs for the right wheels)

---

## Installation

```bash
git clone https://github.com/<your-username>/Ray-trax-3D
cd Ray-trax-3D

# (recommended) isolated env
python -m venv .venv && source .venv/bin/activate
# or: conda create -n raytrax3d python=3.10 -y && conda activate raytrax3d

pip install --upgrade pip
pip install jax jaxlib numpy matplotlib
# GPU users: install the jax/jaxlib wheels matching your CUDA/ROCm toolkit (see JAX docs)
```

---

## Quickstart (3D, time-dependent)

Minimal example using the core time-step kernels (single source and multiple sources). The names below match the public functions in the 3D time-dependent module.

```python
import jax
import jax.numpy as jnp
import numpy as np

from ray_trax_3D_tdep import (
    compute_radiation_field_from_source_with_time_step,
    compute_radiation_field_from_multiple_sources_with_time_step,
)

# 3D regular grid
Nx = Ny = Nz = 128
kappa = jnp.ones((Nx, Ny, Nz)) * 0.3           # opacity field κ(x)
j_map = jnp.zeros_like(kappa)                  # emissivity j(x)
src = (Nx//2, Ny//2, Nz//2)                    # single source at center
j_map = j_map.at[src].set(1.0)                 # point-like emissivity spike

# Marching controls
num_rays            = 4096                     # angular samples
step_size           = 0.5                      # spatial Δs
radiation_velocity  = 1.0                      # c in grid units
time_step           = 1.0                      # Δt => horizon c*Δt
use_sharding        = False                    # set True on multi-GPU nodes

# --- Single-source, one time step ---
J = compute_radiation_field_from_source_with_time_step(
        j_map, kappa, src,
        num_rays=num_rays,
        step_size=step_size,
        radiation_velocity=radiation_velocity,
        time_step=time_step,
        use_sharding=use_sharding
    )

# --- Multiple sources (example) ---
sources = jnp.array([
    (Nx*0.25, Ny*0.50, Nz*0.50),
    (Nx*0.75, Ny*0.50, Nz*0.50),
], dtype=jnp.float32)

J_multi = compute_radiation_field_from_multiple_sources_with_time_step(
    j_map, kappa, sources,
    num_rays=num_rays,
    step_size=step_size,
    radiation_velocity=radiation_velocity,
    time_step=time_step,
    use_sharding=use_sharding
)

# Quick visualization (central Z-slice, log-scaled)
import matplotlib.pyplot as plt
mid = Nz // 2
plt.imshow(np.log10(np.array(J_multi[:, :, mid]) + 1e-6), origin="lower", cmap="inferno")
plt.colorbar()
plt.title("log10 Intensity, Z-mid slice")
plt.tight_layout()
plt.show()
```

**Multi-GPU tip:** when `use_sharding=True`, choose `num_rays` divisible by `len(jax.devices())` so directions partition cleanly across devices.

---

## API at a glance

```python
compute_radiation_field_from_source_with_time_step(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5,
    max_steps=None,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False
) -> J   # shape = j_map.shape

compute_radiation_field_from_multiple_sources_with_time_step(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False
) -> J_total
```

**Notes**
- `max_steps` (if provided) sets a static iteration count; otherwise it’s inferred from `c * Δt / Δs` and rounded up.
- `source_pos` / `source_positions` use grid coordinates (floats allowed).
- Emissivity lives in `j_map`; you may choose to encode “point sources” directly in `j_map` (as shown), or keep `j_map` for diffuse emission and supply discrete `source_positions`.

---

## Differentiability

Ray-trax 3D is built on JAX primitives and is **differentiable end-to-end**:

- Marching uses `jax.lax.fori_loop` and compiles with `jit`.
- Prefer **static** iteration counts (`max_steps`) for stable reverse-mode transforms.
- For **dynamic horizons** or **low-dimensional controls**, **forward-mode** (`jax.jacfwd`) is ideal.
- Symmetric trilinear **interpolate/deposit** keeps the operator smooth in voxel values — essential for stable gradients.

**Example: fit a global amplitude \(A\) via autodiff**
```python
import jax, jax.numpy as jnp

def run(A, j_base, kappa, src, **kw):
    J = compute_radiation_field_from_source_with_time_step(
        j_base * A, kappa, src, **kw
    )
    return J

def mse_loss(A, j_base, kappa, src, J_ref, **kw):
    J = run(A, j_base, kappa, src, **kw)
    return jnp.mean((J - J_ref)**2)

grad_A = jax.grad(mse_loss)(1.0, j_map, kappa, src, J_ref,
                            num_rays=4096, step_size=0.5,
                            radiation_velocity=1.0, time_step=1.0)
```

**Tips**
- Keep arrays on-device and rely on JAX control-flow (`lax.cond`, `lax.scan`) rather than Python branches.
- Avoid in-place mutations; use `.at[...].set/add`.
- For reproducibility: `jax.config.update("jax_enable_x64", True)` where needed.

---

## Testing

Run the full test suite (unit + integration + gradient checks):

```bash
pytest -q
```

What’s covered (typical set):
- **Sampling:** Fibonacci/golden-angle directions (shape & coverage invariants).
- **Interpolation/Deposition:** trilinear consistency and conservation checks.
- **Marching (E2E):** analytic sanity tests on simple media.
- **Autodiff:** numerical vs. analytic gradients for global controls (e.g., amplitude/opacity scale).

**Practical flags**
- Large GPU tests: `XLA_PYTHON_CLIENT_MEM_FRACTION=.85` to cap memory.
- Finite-difference checks: step sizes ~ `1e-3`–`1e-2` work well for bandwidth-bound kernels.

---

## Performance & scaling

- Work scales as \(\mathcal{O}(N_{\text{src}}\,N_\Omega\,N_s)\), with \(N_\Omega\) directions and \(N_s \approx \lceil c\,\Delta t/\Delta s\rceil\).
- Kernels are highly vectorized and typically **bandwidth-bound** on GPUs.
- Treat **frequency/angle bins as batch axes** to increase throughput without branching.
- On multi-GPU nodes, **shard rays** across devices and do a final all-reduce of the accumulator.

> Example result line (for papers):  
> *“On a \(128^3\) grid, tracing \(4{,}096\) rays from \(2{,}097\) sources across the full domain completes in \(X\,\mathrm{s}\) wall-clock on a single NVIDIA A100.”*

---

## FAQ / Troubleshooting

- **`nan`/`inf` gradients or outputs.**  
  Reduce `step_size`, clamp denominators, and inspect with `log10(J + 1e-6)` slices.

- **Slow first call.**  
  That’s XLA compilation. Keep shapes (grid, steps, batch sizes) **static** across runs.

- **GPU OOM.**  
  Decrease `num_rays` or `max_steps`, shard across devices, and minimize host↔device transfers.

---

## License

TBD (research/alpha).
