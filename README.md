# Ray-trax 3D

<p align="right">
  <img src="logo_ray-trax.jpeg" alt="Ray-trax logo" width="140">
</p>

**Ray-trax** is a GPU-oriented, fully differentiable **3D** ray tracer in JAX for the **time-dependent** emission–absorption equation on regular grids. It’s designed for on-the-fly radiative transfer in turbulent astrophysical flows, with clean vectorization across rays, sources, and frequency bins.


---

## Paper

**Ray-trax: Fast, Time-Dependent, and Differentiable Ray Tracing for On-the-Fly Radiative Transfer in Turbulent Astrophysical Flows**  
arXiv: **2511.09389** — https://arxiv.org/abs/2511.09389

---

## Overview

We solve the time-dependent transport of specific intensity \(I(\mathbf{x},\hat{\mathbf{n}},t)\):
\[
\frac{1}{c}\,\frac{\partial I}{\partial t}+\hat{\mathbf n}\!\cdot\!\nabla I
= -\,\kappa(\mathbf{x})\,I + j(\mathbf{x}) ,
\]
with \(\kappa(\mathbf{x})\) (opacity) and \(j(\mathbf{x})\) (emissivity) defined on a regular 3D grid.  
Each time step advances rays over a horizon \(c\,\Delta t\) using a semi-analytic update (exact attenuation over \(\Delta s\), first-order quadrature for emission). We use **Fibonacci/golden-angle** directions, **trilinear interpolation + symmetric deposition** for smooth gradients, and JAX **jit/vmap** (optionally sharded) for speed.

---

## Features

- **Time-dependent, 3D** ray marching; multi-source and multi-bin ready.  
- **Multi-frequency & multi-species** opacities (HI/HeI/HeII photoionization) via a differentiable `kappa` map — enables field-level inverse problems.  
- **Nearly uniform** directional sampling on \(\mathbb{S}^2\) (Fibonacci lattice).  
- **Vectorization** over rays/sources/bins; optional **multi-GPU sharding**.  
- **Differentiable** end-to-end (reverse-mode with static loops; forward-mode for dynamic horizons).  
- **Predictable scaling:** work \(\mathcal{O}(N_{\text{src}}\,N_\Omega\,N_s)\).

---

## Requirements

- Python **3.10+**  
- `jax`, `jaxlib`  
- `numpy`, `matplotlib`

> **GPU users:** install `jax/jaxlib` wheels **matching your CUDA/ROCm** (see JAX docs).

---

## Installation

```bash
git clone https://github.com/<your-username>/Ray-trax-3D
cd Ray-trax-3D

# (recommended) venv or conda
python -m venv .venv && source .venv/bin/activate
# conda create -n raytrax3d python=3.10 -y && conda activate raytrax3d

pip install --upgrade pip
pip install jax jaxlib numpy matplotlib
# GPU wheels: follow the official JAX wheel selector for your CUDA/ROCm
```

---

## Quickstart (3D, time-dependent)

```python
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Run from the repository root (or add it to PYTHONPATH); the solvers
# share primitives in ray_trax/core.py, so import them as a package:
from ray_trax.ray_trax_3D_tdep import (
    compute_radiation_field_from_source_with_time_step,
    compute_radiation_field_from_multiple_sources_with_time_step,
)

# Grid
Nx = Ny = Nz = 128
kappa = jnp.ones((Nx, Ny, Nz)) * 0.3
j_map = jnp.zeros_like(kappa)

# Single source at center (as an emissivity spike)
src = (Nx//2, Ny//2, Nz//2)
j_map = j_map.at[src].set(1.0)

# Marching controls
num_rays = 4096          # angular samples
step_size = 0.5          # spatial Δs
c = 1.0                  # radiation speed in grid units
dt = 1.0                 # time step -> horizon c*dt
use_sharding = False     # set True if using multi-GPU

# Single source
J = compute_radiation_field_from_source_with_time_step(
    j_map, kappa, src,
    num_rays=num_rays,
    step_size=step_size,
    radiation_velocity=c,
    time_step=dt,
    use_sharding=use_sharding
)

# Multiple sources (example)
sources = jnp.array([
    (Nx*0.25, Ny*0.50, Nz*0.50),
    (Nx*0.75, Ny*0.50, Nz*0.50),
], dtype=jnp.float32)

J_multi = compute_radiation_field_from_multiple_sources_with_time_step(
    j_map, kappa, sources,
    num_rays=num_rays,
    step_size=step_size,
    radiation_velocity=c,
    time_step=dt,
    use_sharding=use_sharding
)

# Visualize (central Z-slice, log scale)
mid = Nz // 2
plt.imshow(np.log10(np.array(J_multi[:, :, mid]) + 1e-6),
           origin="lower", cmap="inferno")
plt.colorbar(label="log10(Intensity)")
plt.title("Ray-trax 3D: Z-mid slice")
plt.tight_layout()
plt.show()
```

**Multi-GPU tip:** pick `num_rays` divisible by `len(jax.devices())` for clean sharding.

---

## API at a Glance

```python
compute_radiation_field_from_source_with_time_step(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5,
    max_steps=None,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False
) -> J  # shape = j_map.shape

compute_radiation_field_from_multiple_sources_with_time_step(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False
) -> J_total
```

Notes:
- If `max_steps=None`, we use \(N_s=\lceil c\,\Delta t/\Delta s\rceil\) (static in practice).  
- `source_pos` / `source_positions` accept float grid coords.  
- Encode “point sources” as spikes in `j_map` or supply explicit positions.

---

## Multi-frequency & multi-species opacities

For spectral transport, `ray_trax_3D_multifreq.py` extends the solver with a trailing
frequency axis: `j_map`, `kappa_map`, and the output `J` all have shape
\((N_x,N_y,N_z,n_\text{freq})\). Frequency bins are transported **independently**
(no scattering between bins); inside each ray step the intensity and optical depth are
\((n_\text{freq},)\) vectors updated elementwise, so bins act as a batch axis at
essentially no extra branching cost.

Opacities are built from **photoionization cross sections** (`cross_sections.py`) for
**HI, HeI, HeII**, each with its own ionization threshold:

| species | threshold | \(\sigma\) at threshold | shape |
|---|---|---|---|
| HI   | 13.598 eV | \(6.30\times10^{-18}\,\mathrm{cm^2}\) | \(\propto\nu^{-3}\) |
| HeI  | 24.587 eV | \(7.42\times10^{-18}\,\mathrm{cm^2}\) | two-power-law (Abel+97) |
| HeII | 54.418 eV | \(1.58\times10^{-18}\,\mathrm{cm^2}\) | hydrogenic, \(Z=2\) |

`build_kappa_map` assembles the frequency-resolved opacity from the species densities:
\[
\kappa_\nu(\mathbf{x}) = n_\mathrm{HI}\,\sigma_\mathrm{HI}(\nu)
                        + n_\mathrm{HeI}\,\sigma_\mathrm{HeI}(\nu)
                        + n_\mathrm{HeII}\,\sigma_\mathrm{HeII}(\nu).
\]

```python
import numpy as np, jax.numpy as jnp
from ray_trax.ray_trax_3D_multifreq import compute_radiation_field_multifreq
from ray_trax.cross_sections import build_kappa_map, NU_HI

Nx = Ny = Nz = 64
n_freq  = 6
nu_bins = jnp.array(np.geomspace(NU_HI, 5*NU_HI, n_freq))   # spans HI, HeI, HeII edges

n_HI   = jnp.ones((Nx, Ny, Nz))
zeros  = jnp.zeros((Nx, Ny, Nz))
kappa  = build_kappa_map(nu_bins, n_HI, zeros, zeros)       # (Nx,Ny,Nz,n_freq)

j_map  = jnp.zeros((Nx, Ny, Nz, n_freq)).at[Nx//2, Ny//2, Nz//2, :].set(1.0)
src    = jnp.array([Nx//2, Ny//2, Nz//2], dtype=jnp.float32)

J = compute_radiation_field_multifreq(
        j_map, kappa, src,
        num_rays=8000, step_size=0.5, max_steps=320,
        kappa_interp="trilinear",   # differentiable w.r.t. kappa/n_HI ("nearest" for production)
    )                                # -> J : (Nx, Ny, Nz, n_freq)
```

Because the forward map is differentiable w.r.t. the species densities, the
multi-frequency field enables **field-level inverse problems**: recovering
\(n_\mathrm{HI}\) — or jointly \(n_\mathrm{HI}, n_\mathrm{HeI}, n_\mathrm{HeII}\) — from an
observed \(J(\mathbf{x},\nu)\). The differing thresholds and spectral slopes make the
frequency axis informative enough to **disentangle the three species**
(see `inference_field_multispecies.py`; `inference_field*.py` for the single-species and
background-illumination variants).

---

## Differentiability

- Marching loop uses `jax.lax.fori_loop`, **jit** compiled.  
- Prefer **static** iteration counts for reverse-mode transforms.  
- For **dynamic horizons** or low-dimensional controls, use **forward-mode** (`jax.jacfwd`).  
- **Trilinear interpolate/deposit** keeps the operator smooth in voxel values → stable gradients.

**Example: fit a global amplitude \(A\)**
```python
import jax, jax.numpy as jnp

def forward(A, j_base, kappa, src, **kw):
    return compute_radiation_field_from_source_with_time_step(
        j_base * A, kappa, src, **kw
    )

def loss(A, j_base, kappa, src, J_ref, **kw):
    J = forward(A, j_base, kappa, src, **kw)
    return jnp.mean((J - J_ref)**2)

grad_A = jax.grad(loss)(1.0, j_map, kappa, src, J,
                        num_rays=4096, step_size=0.5,
                        radiation_velocity=1.0, time_step=1.0)
```

---

## Testing

- **Directions:** Fibonacci coverage invariants.  
- **Interp/Deposit:** trilinear consistency & conservation checks.  
- **E2E marching:** analytics in uniform media.  
- **Autodiff:** numerical vs analytic gradients for global controls.

Hints:
- Cap device memory: `XLA_PYTHON_CLIENT_MEM_FRACTION=.85`  
- Finite-difference checks: steps ~`1e-3`–`1e-2`.

---

## Performance & Scaling

- Work: \(\mathcal{O}(N_{\text{src}}\,N_\Omega\,N_s)\), with \(N_s\!\approx\!\lceil c\,\Delta t/\Delta s\rceil\).  
- Highly vectorized; often **bandwidth-bound** on GPUs.  
- Treat **bins** as batch axes to increase throughput without branching.  
- On multi-GPU nodes, **shard rays**; final all-reduce of the accumulator.

> Example line for papers:  
> *On a \(128^3\) grid, tracing \(4{,}096\) rays from \(2{,}097\) sources across the full domain completes in \(X\,\mathrm{s}\) on a single NVIDIA A100.*

---

## Citation

If you use this code, please cite:

- **Ray-trax (arXiv preprint):**  
  *Ray-trax: Fast, Time-Dependent, and Differentiable Ray Tracing for On-the-Fly Radiative Transfer in Turbulent Astrophysical Flows*, arXiv:2511.09389.  
  https://arxiv.org/abs/2511.09389

**BibTeX**
```bibtex
@article{Raytrax2025,
  title   = {Ray-trax: Fast, Time-Dependent, and Differentiable Ray Tracing for On-the-Fly Radiative Transfer in Turbulent Astrophysical Flows},
  author  = {Branca, Lorenzo and Rost, Rune and Buck, Tobias},
  journal = {arXiv preprint arXiv:2511.09389},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.09389}
}
```

---

## License

TBD (research preview).

