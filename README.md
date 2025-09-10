# Ray-trax

**Ray tracing utilities for astrophysical radiative transfer in JAX.**  
Focus on (quasi-)static problems; early **3D** features and **time-dependent** prototypes are included.

> **Status:** research/alpha — interfaces and outputs may change.

---

## Table of contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Examples](#examples)
- [Testing](#testing)
- [Performance](#performance)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Overview

**Ray-trax** provides lightweight, JAX-first building blocks for ray tracing in astrophysical radiative transfer. The design goals are:

- **NumPy-like ergonomics** (write code that looks like `numpy`, runs with `jax.numpy`);
- **Composable primitives** (rays, media, integrators);
- **Accelerated execution** via JIT compilation on CPU/GPU/TPU.

The repository includes working 2D demos, early 3D experiments (including a spatial radiation **direction field**), turbulence/log-normal media generators, and time-dependent prototypes. Figures produced by the examples are saved into `plots*` directories.

---

## Features

- **2D ray tracing** with analytic sanity checks.
- **3D (early)** demos and direction-field visualization.
- **Time-dependent prototypes** for evolving media.
- **Utility scripts** for generating turbulent/log-normal density fields.
- **Plotting helpers** to quickly inspect solutions.

---

## Requirements

- Python **3.10+**
- [`jax`](https://github.com/google/jax) / `jaxlib`
- `numpy`, `matplotlib`
- (Optional) CUDA/ROCm-enabled `jaxlib` for GPU acceleration

---

## Installation

Clone and install dependencies (there is currently no published wheel):

```bash
git clone https://github.com/lorenzobranca/Ray-trax
cd Ray-trax

# (recommended) create an isolated environment
python -m venv .venv && source .venv/bin/activate
# or: conda create -n raytrax python=3.10 -y && conda activate raytrax

# install CPU dependencies
pip install --upgrade pip
pip install jax jaxlib numpy matplotlib

# for GPU: install the appropriate jax/jaxlib wheel for your CUDA/ROCm setup (see JAX docs)

