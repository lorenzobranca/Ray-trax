import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ray_trax.ray_trax_3D import compute_radiation_field_from_source
from ray_trax.utils import gaussian_emissivity  # needs to support 3D version
import time
import os

# Parameters
Nx, Ny, Nz = 64, 64 , 64
kappa0 = 0.1
source_pos = jnp.array([Nx / 2, Ny / 2, Nz / 2])
L = 1.0  # total luminosity for comparison

# === Numerical Setup ===
kappa = jnp.ones((Nx, Ny, Nz)) * kappa0

# Normalized 3D emissivity field (Gaussian)
raw_emissivity = gaussian_emissivity(Nx, Ny, Nz, center=source_pos, amplitude=1.0, width=2.0)
total_raw = jnp.sum(raw_emissivity)
emissivity = raw_emissivity * (L / total_raw)  # Normalize total emissivity to L

# Compute numerical radiation field
start = time.time()
J_numeric = compute_radiation_field_from_source(
    emissivity,
    kappa,
    source_pos=source_pos,
    num_rays=2000,
    step_size=0.5,
    max_steps=400
)
print(f"Ray tracing done in {time.time() - start:.2f} s")

# === Analytical Solution (Spherical symmetry in 3D) ===
X, Y, Z = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), jnp.arange(Nz), indexing='ij')
r = jnp.sqrt((X - source_pos[0])**2 + (Y - source_pos[1])**2 + (Z - source_pos[2])**2)
J_analytic = (L / (4 * jnp.pi * (r + 1e-2)**2)) * jnp.exp(-kappa0 * r)

# Print central value comparison
ix, iy, iz = int(Nx / 2), int(Ny / 2), int(Nz / 2)
print("J_analytic[center] =", J_analytic[ix, iy, iz])
print("J_numeric[center] =", J_numeric[ix, iy, iz])

# === Slices for visualization ===
os.makedirs("plots_3d", exist_ok=True)

for plane, index in zip(["x", "y", "z"], [ix, iy, iz]):
    if plane == "x":
        j_num_slice = J_numeric[index, :, :]
        j_an_slice = J_analytic[index, :, :]
    elif plane == "y":
        j_num_slice = J_numeric[:, index, :]
        j_an_slice = J_analytic[:, index, :]
    else:
        j_num_slice = J_numeric[:, :, index]
        j_an_slice = J_analytic[:, :, index]

    rel_err = jnp.abs((j_num_slice - j_an_slice) / (j_an_slice + 1e-10))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(jnp.log10(j_num_slice + 1e-10), origin='lower', cmap='inferno')
    plt.title(f"Numeric log10(J), {plane}-slice")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(jnp.log10(j_an_slice + 1e-10), origin='lower', cmap='inferno')
    plt.title(f"Analytic log10(J), {plane}-slice")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(jnp.log10(rel_err + 1e-10), origin='lower', cmap='magma')
    plt.title("Relative error log10(|Jnum-Jan|/Jan)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"plots_3d/{plane}_slice.png")
    plt.close()

# === Optional: radial profile
r_vals = r[ix, :, iz]  # a radial line
J_num_line = J_numeric[ix, :, iz]
J_an_line = J_analytic[ix, :, iz]

plt.figure(figsize=(6, 4))
plt.plot(r_vals, J_num_line, label="Numeric")
plt.plot(r_vals, J_an_line, label="Analytic")
plt.yscale('log')
plt.xlabel("Radius r")
plt.ylabel("Intensity J(r)")
plt.legend()
plt.tight_layout()
plt.savefig("plots_3d/radial_profile_line.png")
plt.close()

