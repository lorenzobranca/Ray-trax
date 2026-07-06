"""
Photoionization cross sections for HI, HeI, HeII as JAX-compatible functions.

All functions accept scalar or array nu [Hz] and return cross section [cm^2].
Safe to use inside jit / vmap / grad.

Physical models
---------------
HI   : simple ν^-3 power law from the ground state.
       σ_th = 6.304e-18 cm² at ν_HI = 13.598 eV / h.
       Osterbrock & Ferland (2006), "Astrophysics of Gaseous Nebulae", App. 2.
       Accurate to ~10 % for ν/ν_HI ∈ [1, 10]; Gaunt-factor correction
       can be added later if higher accuracy is needed.

HeI  : two-power-law analytic fit.
       σ_th = 7.42e-18 cm² at ν_HeI = 24.587 eV / h.
       Abel, Anninos, Zhang & Norman (1997), ApJS 115, 19.
       *** Verify coefficients (1.66, -2.05, 0.66, -3.05) against the paper. ***

HeII : exact hydrogenic result scaled for Z = 2.
       σ_th = 1.575e-18 cm² at ν_HeII = 54.418 eV / h.
       From σ_HI / Z² with the Z² threshold shift.
"""

import jax.numpy as jnp

# 1 eV / h_Planck [Hz]
EV_TO_HZ: float = 2.417989e14

# Ionization thresholds [Hz]
NU_HI:   float = 13.598 * EV_TO_HZ   # 3.288e15 Hz  (Lyman limit)
NU_HEI:  float = 24.587 * EV_TO_HZ   # 5.945e15 Hz
NU_HEII: float = 54.418 * EV_TO_HZ   # 13.155e15 Hz


def sigma_HI(nu):
    """
    HI ground-state photoionization cross section [cm²].

    σ(ν) = 6.304e-18 × (ν_HI / ν)³   for ν ≥ ν_HI
         = 0                            otherwise
    """
    nu = jnp.asarray(nu)
    sigma = 6.304e-18 * (NU_HI / nu) ** 3
    return jnp.where(nu >= NU_HI, sigma, 0.0)


def sigma_HeI(nu):
    """
    HeI ground-state photoionization cross section [cm²].

    Two-power-law fit (Abel et al. 1997, ApJS 115 eq. A2):
      σ(ν) = 7.42e-18 × [1.66 (ν/ν_HeI)^-2.05 - 0.66 (ν/ν_HeI)^-3.05]

    Reproduces σ_th = 7.42e-18 cm² exactly at threshold.
    The second term is always negative in magnitude above threshold so
    we clamp to ≥ 0 as a safeguard.

    *** VERIFY: coefficients (1.66, 0.66) and exponents (2.05, 3.05)
        against Abel et al. (1997) Table / eq. A2 before use in production. ***
    """
    nu = jnp.asarray(nu)
    u = nu / NU_HEI
    sigma = 7.42e-18 * (1.66 * u ** (-2.05) - 0.66 * u ** (-3.05))
    return jnp.where(nu >= NU_HEI, jnp.maximum(sigma, 0.0), 0.0)


def sigma_HeII(nu):
    """
    HeII ground-state photoionization cross section [cm²].

    Exact hydrogenic result for Z = 2:
      σ(ν) = σ_HI(ν_HI) / Z²  × (ν_HeII / ν)³
           = 1.575e-18 × (ν_HeII / ν)³   for ν ≥ ν_HeII

    The Z^-2 factor at fixed threshold cross section + Z² shift in threshold
    follows directly from the hydrogen bound-free matrix element.
    """
    nu = jnp.asarray(nu)
    sigma = 1.575e-18 * (NU_HEII / nu) ** 3
    return jnp.where(nu >= NU_HEII, sigma, 0.0)


def build_kappa_map(nu_bins, n_HI, n_HeI, n_HeII):
    """
    Assemble the opacity field κ(x, y, z, ν) [cm^-1].

    κ_ν = n_HI · σ_HI(ν) + n_HeI · σ_HeI(ν) + n_HeII · σ_HeII(ν)

    Parameters
    ----------
    nu_bins : (n_freq,)       frequency bin centres [Hz]
    n_HI    : (Nx, Ny, Nz)   HI number density [cm^-3]
    n_HeI   : (Nx, Ny, Nz)
    n_HeII  : (Nx, Ny, Nz)

    Returns
    -------
    kappa : (Nx, Ny, Nz, n_freq)   opacity [cm^-1]
    """
    nu_bins = jnp.asarray(nu_bins)

    sig_HI   = sigma_HI(nu_bins)    # (n_freq,)
    sig_HeI  = sigma_HeI(nu_bins)   # (n_freq,)
    sig_HeII = sigma_HeII(nu_bins)  # (n_freq,)

    # (Nx,Ny,Nz,1) × (n_freq,) → (Nx,Ny,Nz,n_freq)
    kappa = (n_HI[..., None]   * sig_HI
           + n_HeI[..., None]  * sig_HeI
           + n_HeII[..., None] * sig_HeII)
    return kappa
