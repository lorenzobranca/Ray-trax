import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ray_trax.ray_trax_3D_tdep import compute_radiation_field_from_source_with_time_step
from ray_trax.utils import gaussian_emissivity  # must support 3D: (Nx,Ny,Nz, center, amplitude, width)

# ----------------------- domain & parameters -----------------------
Nx, Ny, Nz = 64, 64, 64
kappa0 = 0.1
L_total = 1.0                      # total luminosity across all sources
num_sources = 10
width = 1.5                        # Gaussian width in grid units
num_rays = 8192
step_size = 0.5
c = 1.0
dt = 50.0

# Absorption field (uniform here)
kappa = jnp.ones((Nx, Ny, Nz), dtype=jnp.float32) * kappa0

# ----------------------- source placement -------------------------
# Example: place sources on a small 3D grid around the center (repeatable pattern)
cx, cy, cz = Nx / 2, Ny / 2, Nz / 2
offsets = jnp.array([
    [-8, -8, -8], [ 8, -8, -8], [-8,  8, -8], [ 8,  8, -8],
    [-8, -8,  8], [ 8, -8,  8], [-8,  8,  8], [ 8,  8,  8],
    [ 0,  0,  0], [12,  0,  0],
], dtype=jnp.float32)
source_positions = jnp.stack([jnp.array([cx, cy, cz], dtype=jnp.float32) + off for off in offsets], axis=0)
print("source positions: ", source_positions)

assert source_positions.shape == (num_sources, 3)

# Per-source luminosities (equal split here)
L_i = jnp.ones((num_sources,), dtype=jnp.float32) * (L_total / num_sources)

# ----------------------- emissivity field -------------------------
# Build a SINGLE emissivity map with 10 Gaussian blobs; each normalized to its L_i
emissivity = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32)
for i in range(num_sources):
    raw = gaussian_emissivity(Nx, Ny, Nz, center=source_positions[i], amplitude=1., width=width)
    raw_sum = jnp.sum(raw)
    # normalize this blob to integrate to L_i
    blob = raw * (L_i[i] / (raw_sum + 1e-20))
    emissivity = emissivity + blob

# ----------------------- ray tracing (sum of sources) -------------
# We your single-source forward once per source and sum.
# NOTE: compute_radiation_field_from_source_with_time_step already scales rays by 4π/num_rays.
start = time.time()
J_sum = jnp.zeros_like(emissivity)
for i in range(num_sources):
    J_i = compute_radiation_field_from_source_with_time_step(
        emissivity,
        kappa,
        source_pos=source_positions[i],
        num_rays=int(num_rays),
        step_size=float(step_size),
        radiation_velocity=float(c),
        time_step=float(dt),
        use_sharding=False,      # set True if you’ve configured sharding
    )
    J_sum = J_sum + J_i
elapsed = time.time() - start
print(f"Ray tracing for {num_sources} sources done in {elapsed:.2f} s")

# ----------------------- analytic comparison ----------------------
# Sum of 10 point-source terms: J(r) = Σ_i [ L_i / (4π r_i^2) * exp(-kappa0 * r_i) ]
X, Y, Z = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), jnp.arange(Nz), indexing='ij')

def point_source_field(center, Lsrc):
    dx = X - center[0]; dy = Y - center[1]; dz = Z - center[2]
    r = jnp.sqrt(dx*dx + dy*dy + dz*dz)
    return (Lsrc / (4.0 * jnp.pi * (r + 1e-6 )**2)) * jnp.exp(-kappa0 * (r))

J_analytic = jnp.zeros_like(J_sum)
for i in range(num_sources):
    J_analytic = J_analytic + point_source_field(source_positions[i], L_i[i])



ix, iy, iz = int(Nx/2), int(Ny/2), int(Nz/2)
print("J_analytic[center] =", float(J_analytic[ix, iy, iz]))
print("J_numeric[center]  =", float(J_sum[ix, iy, iz]))

# source_positions is shape (num_sources, 3), float
idx = source_positions.astype(int)  # integer indices (num_sources, 3)

# gather the values from J_analytic at those positions
val = J_sum[idx[:, 0], idx[:, 1], idx[:, 2]]

# scatter them into J_sum
J_analytic = J_analytic.at[idx[:, 0], idx[:, 1], idx[:, 2]].set(val)

print("J_analytic[center] =", float(J_analytic[ix, iy, iz]))
print("J_numeric[center]  =", float(J_sum[ix, iy, iz]))


# ----------------------- plotting: slices -------------------------
os.makedirs("plots_3d_multi", exist_ok=True)

def save_slice(plane, index):
    if plane == "x":
        num_slice = J_sum[index, 1:-1, 1:-1]
        an_slice  = J_analytic[index, 1:-1, 1:-1]
    elif plane == "y":
        num_slice = J_sum[1:-1, index, 1:-1]
        an_slice  = J_analytic[1:-1, index, 1:-1]
    else:
        num_slice = J_sum[1:-1, 1:-1, index]
        an_slice  = J_analytic[1:-1, 1:-1, index]

    rel_err = jnp.abs((num_slice - an_slice) / (an_slice + 1e-10))
    print("rel err:", jnp.mean(rel_err[1:-1, 1:-1]))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(jnp.log10(num_slice + 1e-10), origin='lower', cmap='inferno'); plt.title(f"Numeric log10(J), {plane}-slice"); plt.colorbar()
    plt.subplot(1, 3, 2); plt.imshow(jnp.log10(an_slice  + 1e-10), origin='lower', cmap='inferno'); plt.title(f"Analytic log10(J), {plane}-slice"); plt.colorbar()
    plt.subplot(1, 3, 3); plt.imshow(jnp.log10(rel_err + 1e-10), origin='lower', cmap='magma');  plt.title("Relative error log10(|Δ|/J_an)");  plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"plots_3d_multi/slice_{plane}.png"); plt.close()


save_slice("x", ix)
save_slice("y", iy)
save_slice("z", iz)

# ----------------------- optional: a radial line -------------------
# choose one source and plot along a line passing through it
j = 0
sx, sy, sz = map(int, map(float, source_positions[j]))
r_line = jnp.sqrt((X[sx, 1:-1, sz] - source_positions[j,0])**2 + (Y[sx, 1:-1, sz] - source_positions[j,1])**2)
J_num_line = J_sum[sx, 1:-1, sz]
J_an_line  = J_analytic[sx, 1:-1, sz]

plt.figure(figsize=(6, 4))
plt.plot(r_line, J_num_line, label="Numeric")
plt.plot(r_line, J_an_line, label="Analytic", linestyle="--")
plt.yscale('log'); plt.xlabel("Radius r"); plt.ylabel("Intensity J(r)"); plt.legend()
plt.tight_layout(); plt.savefig("plots_3d_multi/radial_profile_line.png"); plt.close()


# ----------------------- diagonal plane utilities -----------------------
try:
    from scipy.ndimage import map_coordinates
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _trilinear_sample_jax(vol, x, y, z):
    """vol: (Nx,Ny,Nz), x/y/z same shape, index coordinates (0..N-1). Returns sampled values."""
    Nx, Ny, Nz = vol.shape
    # clamp to [0, N-1 - eps] so ceil is in-bounds
    eps = 1e-6
    x = jnp.clip(x, 0.0, Nx - 1 - eps)
    y = jnp.clip(y, 0.0, Ny - 1 - eps)
    z = jnp.clip(z, 0.0, Nz - 1 - eps)

    x0 = jnp.floor(x).astype(jnp.int32); x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32); y1 = y0 + 1
    z0 = jnp.floor(z).astype(jnp.int32); z1 = z0 + 1

    x1 = jnp.clip(x1, 0, Nx - 1)
    y1 = jnp.clip(y1, 0, Ny - 1)
    z1 = jnp.clip(z1, 0, Nz - 1)

    xd = x - x0; yd = y - y0; zd = z - z0

    def gather(a, xi, yi, zi):  # advanced indexing
        return a[xi, yi, zi]

    c000 = gather(vol, x0, y0, z0)
    c100 = gather(vol, x1, y0, z0)
    c010 = gather(vol, x0, y1, z0)
    c110 = gather(vol, x1, y1, z0)
    c001 = gather(vol, x0, y0, z1)
    c101 = gather(vol, x1, y0, z1)
    c011 = gather(vol, x0, y1, z1)
    c111 = gather(vol, x1, y1, z1)

    c00 = c000*(1-xd) + c100*xd
    c10 = c010*(1-xd) + c110*xd
    c01 = c001*(1-xd) + c101*xd
    c11 = c011*(1-xd) + c111*xd

    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd
    return c0*(1-zd) + c1*zd

def _sample_plane(field, origin, u_vec, v_vec, nu, nv):
    """
    Sample a plane from a 3D field using index-space vectors.
    origin: (ox,oy,oz) in index coords
    u_vec, v_vec: direction vectors in index coords (not necessarily unit)
    nu, nv: resolution along u and v
    Returns (plane, Xp, Yp) where plane shape is (nu,nv)
    """
    ox, oy, oz = origin
    u_vec = jnp.asarray(u_vec, dtype=jnp.float32)
    v_vec = jnp.asarray(v_vec, dtype=jnp.float32)

    # choose spans that roughly cover the volume with some margin
    Nx, Ny, Nz = field.shape
    # heuristic spans based on min dimension in projected axes
    span_u = 0.9 * min(Nx, Ny)   # works well for x=y planes
    span_v = 0.9 * Nz            # vertical extent
    u = jnp.linspace(-span_u/2, span_u/2, nu)
    v = jnp.linspace(-span_v/2, span_v/2, nv)
    U, V = jnp.meshgrid(u, v, indexing='ij')

    P = jnp.array([ox, oy, oz], dtype=jnp.float32)[:, None, None]
    UV = (u_vec[:, None, None] * U) + (v_vec[:, None, None] * V)
    pts = P + UV  # shape (3, nu, nv) in index coordinates

    x = pts[0]; y = pts[1]; z = pts[2]

    if _HAS_SCIPY:
        # scipy expects coordinates in (z,y,x) order as floats (but our axes are (x,y,z))
        coords = jnp.stack([z, y, x], axis=0)
        # convert to numpy for scipy
        import numpy as _np
        plane = map_coordinates(_np.asarray(field), _np.asarray(coords), order=1, mode='nearest')
        plane = jnp.asarray(plane)
    else:
        plane = _trilinear_sample_jax(field, x, y, z)

    return plane, x, y  # x,y can be used just for reference if needed

def save_diagonal_plane(name="xy_diag", nu=256, nv=256):
    """
    Saves plots for the diagonal vertical plane x = y through the domain center.
    """
    Nx, Ny, Nz = J_sum.shape
    cx, cy, cz = (Nx-1)/2.0, (Ny-1)/2.0, (Nz-1)/2.0

    origin = (cx, cy, cz)
    u_vec  = jnp.array([1.0, 1.0, 0.0])  # along x=y
    v_vec  = jnp.array([0.0, 0.0, 1.0])  # along +z

    num_plane, _, _ = _sample_plane(J_sum,      origin, u_vec, v_vec, nu, nv)
    an_plane,  _, _ = _sample_plane(J_analytic, origin, u_vec, v_vec, nu, nv)

    rel_err = jnp.abs((num_plane - an_plane) / (an_plane + 1e-10))
    print("diag x=y rel err:", float(jnp.mean(rel_err[1:-1, 1:-1])))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(jnp.log10(num_plane + 1e-10), origin='lower', cmap='inferno'); plt.title("Numeric log10(J), diag x=y"); plt.colorbar()
    plt.subplot(1, 3, 2); plt.imshow(jnp.log10(an_plane  + 1e-10), origin='lower', cmap='inferno'); plt.title("Analytic log10(J), diag x=y"); plt.colorbar()
    plt.subplot(1, 3, 3); plt.imshow(jnp.log10(rel_err + 1e-10), origin='lower', cmap='magma');  plt.title("Relative error log10(|Δ|/J_an)");  plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"plots_3d_multi/diag_plane_{name}.png"); plt.close()

# example calls:
os.makedirs("plots_3d_multi", exist_ok=True)
save_diagonal_plane(name="xy_diag", nu=256, nv=256)

# Optional: other diagonal planes just by changing u_vec/v_vec
def save_custom_diagonal(name, u_vec, v_vec, nu=256, nv=256):
    Nx, Ny, Nz = J_sum.shape
    cx, cy, cz = (Nx-1)/2.0, (Ny-1)/2.0, (Nz-1)/2.0
    origin = (cx, cy, cz)
    num_plane, _, _ = _sample_plane(J_sum,      origin, u_vec, v_vec, nu, nv)
    an_plane,  _, _ = _sample_plane(J_analytic, origin, u_vec, v_vec, nu, nv)
    rel_err = jnp.abs((num_plane - an_plane) / (an_plane + 1e-10))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(jnp.log10(num_plane + 1e-10), origin='lower', cmap='inferno'); plt.title(f"Numeric log10(J), {name}"); plt.colorbar()
    plt.subplot(1, 3, 2); plt.imshow(jnp.log10(an_plane  + 1e-10), origin='lower', cmap='inferno'); plt.title(f"Analytic log10(J), {name}"); plt.colorbar()
    plt.subplot(1, 3, 3); plt.imshow(jnp.log10(rel_err + 1e-10), origin='lower', cmap='magma');  plt.title("Relative error log10(|Δ|/J_an)");  plt.colorbar()
    plt.tight_layout(); plt.savefig(f"plots_3d_multi/diag_plane_{name}.png"); plt.close()

# Examples for other diagonals:
# x = z plane (span along x=z, vertical along y):
# save_custom_diagonal("xz_diag", u_vec=jnp.array([1.0, 0.0, 1.0]), v_vec=jnp.array([0.0, 1.0, 0.0]))
# y = z plane (span along y=z, vertical along x):
# save_custom_diagonal("yz_diag", u_vec=jnp.array([0.0, 1.0, 1.0]), v_vec=jnp.array([1.0, 0.0, 0.0]))

