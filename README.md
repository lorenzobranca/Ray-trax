# Ray-trax 3D
<p align="right">
  <img src="logo_ray-trax.jpeg" alt="Ray-trax logo" width="140">
</p>

**Time-dependent radiative transfer in 3D with JAX** — fast, differentiable ray marching for the monochromatic emission–absorption equation on regular grids. Built for research-grade workloads, GPU acceleration, and gradient-based workflows.

> 3D is the primary, stable target of this repository.

---

## Table of contents
- [Overview](#overview)
- [Paper](#paper)
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

## Paper

If you use Ray-trax in academic work, please cite our paper:

**Ray-trax: Fast, Time-Dependent, and Differentiable Ray Tracing for On-the-Fly Radiative Transfer in Turbulent Astrophysical Flows**  
arXiv: **2511.09389** — https://arxiv.org/abs/2511.09389

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

