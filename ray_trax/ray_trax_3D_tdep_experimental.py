import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec as P
import optax
from dataclasses import dataclass

def sample_sphere(n):
    i = jnp.arange(0, n)
    golden_ratio = (1 + jnp.sqrt(5)) / 2
    phi = 2 * jnp.pi * i / golden_ratio
    cos_theta = 1 - 2 * (i + 0.5) / n
    sin_theta = jnp.sqrt(1 - cos_theta**2)
    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = cos_theta
    return jnp.stack([x, y, z], axis=1)


def trilinear_op(grid, x, y, z, value=None, mode="interp"):
    Nx, Ny, Nz = grid.shape
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    z0 = jnp.floor(z).astype(jnp.int32)

    x1 = jnp.clip(x0 + 1, 0, Nx - 1)
    y1 = jnp.clip(y0 + 1, 0, Ny - 1)
    z1 = jnp.clip(z0 + 1, 0, Nz - 1)
    x0 = jnp.clip(x0, 0, Nx - 1)
    y0 = jnp.clip(y0, 0, Ny - 1)  # <-- fixed
    z0 = jnp.clip(z0, 0, Nz - 1)

    dx, dy, dz = (x - x0), (y - y0), (z - z0)

    w000 = (1-dx)*(1-dy)*(1-dz); w001 = (1-dx)*(1-dy)*dz
    w010 = (1-dx)*dy*(1-dz);     w011 = (1-dx)*dy*dz
    w100 = dx*(1-dy)*(1-dz);     w101 = dx*(1-dy)*dz
    w110 = dx*dy*(1-dz);         w111 = dx*dy*dz

    if mode == "interp":
        v000 = grid[x0, y0, z0]; v001 = grid[x0, y0, z1]
        v010 = grid[x0, y1, z0]; v011 = grid[x0, y1, z1]
        v100 = grid[x1, y0, z0]; v101 = grid[x1, y0, z1]
        v110 = grid[x1, y1, z0]; v111 = grid[x1, y1, z1]
        return (w000*v000 + w001*v001 + w010*v010 + w011*v011 +
                w100*v100 + w101*v101 + w110*v110 + w111*v111)

    elif mode == "deposit":
        assert value is not None
        out = grid
        out = out.at[x0, y0, z0].add(w000 * value)
        out = out.at[x0, y0, z1].add(w001 * value)
        out = out.at[x0, y1, z0].add(w010 * value)
        out = out.at[x0, y1, z1].add(w011 * value)
        out = out.at[x1, y0, z0].add(w100 * value)
        out = out.at[x1, y0, z1].add(w101 * value)
        out = out.at[x1, y1, z0].add(w110 * value)
        out = out.at[x1, y1, z1].add(w111 * value)
        return out

    else:
        raise ValueError("mode must be 'interp' or 'deposit'")


@partial(jax.jit, static_argnames=[
    "num_rays", "step_size", "use_sharding", "max_steps", "radiation_velocity", "step_size"
])

def compute_radiation_field_from_source_with_time_step(
    j_map, kappa_map, source_pos,
    num_rays=1000, step_size=0.5,
    max_steps = None,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False
):
    """
    Compute the radiation field within one time step (propagation distance = c * dt).
    """
    Nx, Ny, Nz = j_map.shape
    directions = sample_sphere(num_rays)
    
    if max_steps is None:
        max_distance = radiation_velocity * time_step
        max_steps = jnp.ceil(max_distance / step_size).astype(int)


    def trace_single_ray(direction):
        x, y, z = source_pos
        I = 0.0
        tau = 0.0
        J = jnp.zeros_like(j_map)

        def body_fn(i, state):
            x, y, z, I, tau, J = state
            j_val = trilinear_op(j_map, x, y, z, mode="interp")
            #kappa_val = trilinear_op(kappa_map, x, y, z, mode="interp") #maybe issue, see possible fixing below
            ix = jnp.clip(jnp.floor(x).astype(int), 0, Nx - 1)
            iy = jnp.clip(jnp.floor(y).astype(int), 0, Ny - 1)
            iz = jnp.clip(jnp.floor(z).astype(int), 0, Nz - 1)

            kappa_val = kappa_map[ix,iy,iz]

            d_tau = kappa_val * step_size
            dI = j_val * jnp.exp(-tau) * step_size
            I_new = I * jnp.exp(-d_tau) + dI
            tau_new = tau + d_tau
            J = trilinear_op(J, x, y, z, value=I_new, mode="deposit")
            x_new = x + direction[0] * step_size
            y_new = y + direction[1] * step_size
            z_new = z + direction[2] * step_size
            return (x_new, y_new, z_new, I_new, tau_new, J)

        state_init = (x, y, z, I, tau, J)
        final_state = jax.lax.fori_loop(0, max_steps, body_fn, state_init)
        return final_state[-1]  # Return J

    if use_sharding:
        # Build mesh and sharding
        devices = jax.devices()
        n_devices = len(devices)

        if num_rays % n_devices != 0:
            raise ValueError(f"num_rays ({num_rays}) must be divisible by number of devices ({n_devices}).")

        mesh_shape = (n_devices,)
        #mesh = jax.sharding.Mesh(np.array(devices).reshape(mesh_shape), axis_names=('x',))
        mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('x',))
        #sharding = NamedSharding(mesh, P('x', None))  # rays over 'x', each ray is a (3,) vector
        sharding = NamedSharding(mesh, P('x'))

        # Reshape and shard directions
        directions_reshaped = directions.reshape((n_devices, -1, 3))
        directions_sharded = jax.device_put(directions_reshaped, sharding)

        # Apply vmap over each shard
        @jax.vmap  # Automatically parallel within each device
        def trace_ray_batch(dir_batch):
            
            #return jax.vmap(trace_single_ray)(dir_batch)
            return jnp.sum(jax.vmap(trace_single_ray)(dir_batch), axis=0)
            #return jax.lax.psum_scatter(jax.vmap(trace_single_ray)(dir_batch), axis_name="x")

        with mesh:
            J_all_sharded = trace_ray_batch(directions_sharded)  # [n_devices, rays_per_device, Nx, Ny, Nz]

        # Merge all rays
        #J_all = J_all_sharded.reshape((num_rays, *j_map.shape))  # [num_rays, Nx, Ny, Nz]
        J_all = J_all_sharded.reshape((n_devices, *j_map.shape))

    else:
        J_all = jax.vmap(trace_single_ray)(directions)

    return jnp.sum(J_all, axis=0) * (4 * jnp.pi / num_rays)



def compute_radiation_field_from_multiple_sources_with_time_step(
    j_map, kappa_map, source_positions,
    num_rays=1000, step_size=0.5,
    radiation_velocity=1.0, time_step=1.0,
    use_sharding=False
):
    """
    Compute the radiation field within one time step from multiple sources using lax.fori_loop.
    """

    def body_fn(i, I_total):
        source_pos = source_positions[i]
        I_new = compute_radiation_field_from_source_with_time_step(
            j_map, kappa_map, source_pos,
            num_rays=num_rays,
            step_size=step_size,
            radiation_velocity=radiation_velocity,
            time_step=time_step,
            use_sharding=use_sharding
        )
        return I_total + I_new

    num_sources = source_positions.shape[0]
    I_total = jnp.zeros_like(j_map)
    I_total = jax.lax.fori_loop(0, num_sources, body_fn, I_total)

    return I_total


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
