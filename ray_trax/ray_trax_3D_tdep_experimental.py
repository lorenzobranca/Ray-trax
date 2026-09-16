"""
Experimental: variational inference of a log-normal opacity PDF.

The time-dependent solver formerly duplicated here now lives in
ray_trax_3D_tdep (re-exported below for backwards compatibility).

NOTE: `infer_logpdf_kappa_lognormal` still refers to a per-ray forward model
`trace_ray_batch` / `trace_ray_batch_sharded` that is not defined anywhere in
the package, so it is not runnable as-is.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ray_trax.core import sample_sphere  # noqa: F401
from ray_trax.ray_trax_3D_tdep import (  # noqa: F401
    compute_radiation_field_from_source_with_time_step,
    compute_radiation_field_from_multiple_sources_with_time_step,
)


@dataclass
class KappaPDFCfg:
    K: int = 6                 # number of κ samples per step
    lr: float = 5e-2
    steps: int = 400
    clip_norm: float = 1.0
    log_every: int = 25
    use_energy_distance: bool = True  # else moment match

def _energy_distance(a, b):
    # E|A-A'| + E|B-B'| − 2E|A−B|
    def mean_abs_diff(x, y):
        # small subsample for O(N^2) stability if huge
        maxn = 4096
        if x.size > maxn: x = x[jax.random.permutation(jax.random.PRNGKey(0), x.size)[:maxn]]
        if y.size > maxn: y = y[jax.random.permutation(jax.random.PRNGKey(1), y.size)[:maxn]]
        return jnp.mean(jnp.abs(x[:, None] - y[None, :]))
    return mean_abs_diff(a, a) + mean_abs_diff(b, b) - 2.0 * mean_abs_diff(a, b)


def infer_logpdf_kappa_lognormal(
    y_obs: jax.Array,            # [S*R] observed intensities (you can pass a normalized version)
    E_fixed: jax.Array,          # [Nx,Ny,Nz] emissivity (fixed)
    kappa_shape: tuple,          # (Nx,Ny,Nz)
    star_positions: jax.Array,   # [S,3]
    num_rays: int,
    step_size: float,
    c: float,
    dt: float,
    use_sharded: bool = False,
    cfg: KappaPDFCfg = KappaPDFCfg(),
    key: jax.random.Key = jax.random.PRNGKey(0),
):
    Nx, Ny, Nz = map(int, kappa_shape)

    # ---------- host-side numerics (STATIC) ----------
    n_steps_py = max(1, int(np.floor(float(c) * float(dt) / float(step_size))))
    dirs = sample_sphere(int(num_rays))                        # [R,3]
    S = int(star_positions.shape[0])
    origins = jnp.repeat(star_positions, int(num_rays), 0)     # [S*R,3]
    dirs_big = jnp.tile(dirs, (S, 1))                          # [S*R,3]
    trace_fn = trace_ray_batch_sharded if (use_sharded and 'trace_ray_batch_sharded' in globals()) else trace_ray_batch

    @jax.jit
    def fwd_kappa(kappa_field: jax.Array) -> jax.Array:
        # Your exact forward, with κ as the variable
        pred = trace_fn(E_fixed, kappa_field, origins, dirs_big, step_size, n_steps_py)  # [S*R]
        return pred

    # ---------- parameters of log κ ~ N(μ, σ^2) ----------
    # We'll optimize μ (real) and log_sigma (real); sigma = softplus(log_sigma)
    mu = jnp.array(-2.0, jnp.float32)         # start around exp(-2) ~ 0.14
    log_sigma = jnp.array(0.0, jnp.float32)   # sigma starts near softplus(0) ~ 0.693

    opt = optax.chain(
        optax.clip_by_global_norm(cfg.clip_norm),
        optax.adam(cfg.lr),
    )
    opt_state = opt.init((mu, log_sigma))

    def sample_kappas(mu, sigma, k, key):
        # Reparam: log κ = mu + sigma * ξ, ξ~N(0,1); κ = exp(log κ)
        # i.i.d. per voxel; if you want to scale by a base density ρ(x), use kappa = ρ * exp(...)
        keys = jax.random.split(key, k)
        def _one(kk):
            xi = jax.random.normal(kk, (Nx, Ny, Nz), dtype=jnp.float32)
            return jnp.exp(mu + sigma * xi)
        return jax.vmap(_one)(keys)  # [K,Nx,Ny,Nz]

    @jax.jit
    def loss_and_grads(mu, log_sigma, key):
        sigma = jax.nn.softplus(log_sigma)
        k1, k2 = jax.random.split(key)
        kappas = sample_kappas(mu, sigma, cfg.K, k1)            # [K, Nx,Ny,Nz]
        preds = jax.vmap(fwd_kappa)(kappas)                     # [K, S*R]
        preds = preds.reshape(-1)                               # pool samples

        if cfg.use_energy_distance:
            loss_data = _energy_distance(preds, y_obs)
        else:
            # moment matching (mean + variance)
            m_p, v_p = jnp.mean(preds), jnp.var(preds)
            m_o, v_o = jnp.mean(y_obs), jnp.var(y_obs)
            loss_data = (m_p - m_o)**2 + (jnp.sqrt(v_p + 1e-12) - jnp.sqrt(v_o + 1e-12))**2

        # mild priors to keep params reasonable
        loss_reg = 1e-4 * (mu**2) + 1e-4 * (sigma**2)
        loss = loss_data + loss_reg
        return loss, (loss_data, loss_reg, sigma, k2)

    @jax.jit
    def step(mu, log_sigma, opt_state, key):
        (loss, (ldata, lreg, sigma, key_out)), grads = jax.value_and_grad(loss_and_grads, has_aux=True)(mu, log_sigma, key)
        updates, opt_state = opt.update(grads, opt_state, (mu, log_sigma))
        mu_new, log_sigma_new = optax.apply_updates((mu, log_sigma), updates)
        return mu_new, log_sigma_new, opt_state, loss, ldata, lreg, sigma, key_out

    # ---------- training loop ----------
    logs = []
    for it in range(cfg.steps):
        mu, log_sigma, opt_state, L, Ld, Lr, sigma, key = step(mu, log_sigma, opt_state, key)
        if (it % cfg.log_every) == 0 or it == cfg.steps - 1:
            logs.append((float(L), float(Ld), float(Lr), float(mu), float(sigma)))
            print(f"[{it:4d}] loss={float(L):.5e} data={float(Ld):.5e} reg={float(Lr):.2e}  mu={float(mu):+.3f}  sigma={float(sigma):.3f}")

    # return fitted log-PDF params and a callable for κ-PDF on a grid
    mu_f, sigma_f = float(mu), float(jax.nn.softplus(log_sigma))
    def kappa_pdf(kappa_grid: np.ndarray) -> np.ndarray:
        # lognormal pdf with params (mu_f, sigma_f)
        x = np.asarray(kappa_grid, dtype=np.float64)
        eps = 1e-30
        return (1.0 / (x * sigma_f * np.sqrt(2*np.pi) + eps)) * np.exp(-0.5 * ((np.log(x + eps) - mu_f)/sigma_f)**2)

    return (mu_f, sigma_f), logs, kappa_pdf
