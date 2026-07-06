"""
background.py — uniform-background (absorption-tomography) forward model.

Complementary to the point-source ray tracers: instead of launching rays
*outward* from a point (I=0, growing by emission, flux ~1/r^2), a known
isotropic background intensity I_bg enters the box from the boundary and is
*absorbed inward*. The mean intensity at each voxel is the background,
angle-averaged, attenuated by the optical depth accumulated from the boundary
to that voxel.

Why this matters for the inverse problem
----------------------------------------
The point-source map has a ~99% nullspace: sensitivity |dJ/dn_HI| falls off
like 1/r^2, so only the illuminated core carries an absorption signal. A
background field instead crosses *every* voxel along many sightlines, so
sensitivity is ~uniform and the full-box recovery of n_HI becomes well-posed
(this is the Lyα-forest / CT absorption-tomography regime).

Discretization
--------------
The isotropic background is approximated by the 6 axis-aligned beam directions
(±x, ±y, ±z). Along an axis, the attenuation is exact and vectorized: the
optical depth entering cell i is the exclusive cumulative sum of the per-cell
optical depth, so no ray marching is needed. This is differentiable w.r.t.
kappa_map (hence n_HI) everywhere — no nearest-neighbour lookup — and runs in a
handful of cumsum ops. More angular coverage (oblique beams) can be added later
via the ray tracer; 6-axis already flattens the 1/r^2 sensitivity.

Units
-----
kappa_map is the optical depth PER GRID CELL (i.e. already n_HI * sigma * DX_PHYS,
as produced by build_kappa_map(...) * DX_PHYS). Crossing one cell therefore adds
tau = kappa. The output J is the angle-averaged (over the 6 beams) mean intensity.

Inputs
------
kappa_map : (Nx, Ny, Nz, n_freq)   optical depth per cell
I_bg      : (n_freq,)              background intensity per frequency bin

Output
------
J         : (Nx, Ny, Nz, n_freq)   mean background intensity
"""

from functools import partial
import jax
import jax.numpy as jnp


def _beam_along(kappa_map, I_bg, axis, reverse):
    """Intensity entering each cell for a beam travelling along `axis`.

    reverse=False: beam enters at the low-index face and travels +axis.
    reverse=True : beam enters at the high-index face and travels -axis.
    """
    k = jnp.flip(kappa_map, axis=axis) if reverse else kappa_map
    # exclusive cumulative optical depth: tau entering cell i = sum_{i'<i} k[i']
    tau_incl = jnp.cumsum(k, axis=axis)
    tau_excl = tau_incl - k
    I = I_bg * jnp.exp(-tau_excl)             # (..., n_freq), broadcast over I_bg
    return jnp.flip(I, axis=axis) if reverse else I


@partial(jax.jit, static_argnames=("normalize",))
def compute_background_field_multifreq(kappa_map, I_bg, normalize=True):
    """Mean intensity from an isotropic background, 6-axis-beam approximation.

    Parameters
    ----------
    kappa_map : (Nx, Ny, Nz, n_freq)  optical depth per grid cell
    I_bg      : (n_freq,)             background intensity per bin
    normalize : if True, average over the 6 beams (mean intensity J);
                if False, return the summed contribution.

    Returns
    -------
    J : (Nx, Ny, Nz, n_freq)
    """
    I_bg = jnp.asarray(I_bg)
    beams = [
        _beam_along(kappa_map, I_bg, axis, rev)
        for axis in (0, 1, 2)
        for rev in (False, True)
    ]
    J = jnp.stack(beams, axis=0).sum(axis=0)
    return J / len(beams) if normalize else J
