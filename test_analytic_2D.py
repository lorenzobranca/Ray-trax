import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from ray_trax.ray_trax_2D import compute_radiation_field_from_source
from ray_trax.utils import gaussian_emissivity 
import time

# Parameters
Nx, Ny = 200, 200
kappa0 = 0.1
source_pos = jnp.array([Nx / 2, Ny / 2])
L = 1.0  # total luminosity for both numerical and analytical comparison

# === Numerical Setup ===
kappa = jnp.ones((Nx, Ny)) * kappa0

# Scale emissivity so that its total sum equals L
raw_emissivity = gaussian_emissivity(Nx, Ny, center=source_pos, amplitude=1.0, width=0.5)
total_raw = jnp.sum(raw_emissivity)
emissivity = raw_emissivity * (L / total_raw)  # Normalized to have total luminosity L

# Compute numerical radiation field using ray tracing
J_numeric = compute_radiation_field_from_star(
    emissivity,
    kappa,
    source_pos=source_pos,
    num_rays=int(360 * 10),
    step_size=0.5,
    max_steps=400
)

# === Analytical Solution ===
X, Y = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), indexing='ij')
r = jnp.sqrt((X - source_pos[0])**2 + (Y - source_pos[1])**2)
#J_analytic = (L / (4 * jnp.pi * (r + 1e-2)**2)) * jnp.exp(-kappa0 * r)
J_analytic = (L / (2 * jnp.pi * (r + 1e-2))) * jnp.exp(-kappa0 * r)

# Print comparison at center
print("J_analytic[100,100] =", J_analytic[100, 100])
print("J_numeric[100,100] =", J_numeric[100, 100])

# === Plotting ===
plt.figure(figsize=(12, 5))

# Numerical result
plt.subplot(1, 3, 1)
plt.imshow(jnp.log10(J_numeric[1:-1, 1:-1] + 1e-10), origin='lower', cmap='inferno')
plt.title("Ray Traced Log10(J)")
plt.colorbar()

# Analytical result
plt.subplot(1, 3, 2)
plt.imshow(jnp.log10(J_analytic[1:-1, 1:-1] + 1e-10), origin='lower', cmap='inferno')
plt.title("Analytical Log10(J)")
plt.colorbar()

# Relative error
rel_err = jnp.abs((J_numeric - J_analytic) / (J_analytic + 1e-10))
plt.subplot(1, 3, 3)
plt.imshow(jnp.log10(rel_err[1:-1, 1:-1]), origin='lower', cmap='magma')
plt.title("Relative Error Log10(|Jnum - Jtrue| / Jtrue)")
plt.colorbar()

plt.tight_layout()
plt.savefig("plots/num_vs_exact.png")
plt.close()

# === Radial profile comparison ===
r_vals = jnp.sqrt((X[1:-1, 1:-1] - source_pos[0])**2 + (Y[1:-1, 1:-1] - source_pos[1])**2)
r_flat = r_vals.flatten()
J_num_flat = J_numeric[1:-1, 1:-1].flatten()
J_an_flat = J_analytic[1:-1, 1:-1].flatten()

plt.figure(figsize=(6, 4))
plt.scatter(r_flat, J_num_flat, s=2, alpha=0.5, label='Numeric')
plt.scatter(r_flat, J_an_flat, s=2, alpha=0.5, label='Analytic')
plt.yscale('log')
plt.xlabel("Radius r")
plt.ylabel("Intensity J(r)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/radial_profile.png")
plt.close()

