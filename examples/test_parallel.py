"""
Test multi-GPU parallelization using shard_map.
Uses autocvd to safely grab 2 free GPUs.
Verifies that single-GPU and multi-GPU results match, and reports timing.
"""
from autocvd import autocvd
autocvd(num_gpus=2)   # sets CUDA_VISIBLE_DEVICES to 2 free GPUs

import jax
import jax.numpy as jnp
import time

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")
assert len(jax.devices()) == 2, f"Expected 2 GPUs, got {len(jax.devices())}"

from ray_trax.ray_trax_3D import compute_radiation_field_from_source
from ray_trax.ray_trax_3D_tdep import compute_radiation_field_from_source_with_time_step

# --- shared setup ---
Nx, Ny, Nz = 64, 64, 64
source_pos  = jnp.array([Nx / 2.0, Ny / 2.0, Nz / 2.0])
kappa       = jnp.ones((Nx, Ny, Nz)) * 0.1
emissivity  = jnp.zeros((Nx, Ny, Nz)).at[Nx // 2, Ny // 2, Nz // 2].set(1.0)
NUM_RAYS    = 2048   # divisible by 2
STEP_SIZE   = 0.5
MAX_STEPS   = 200
REL_TOL     = 1e-4   # max acceptable relative difference


def timed_call(fn, *args, n_warmup=1, n_bench=3, **kwargs):
    """Warmup then average over n_bench calls."""
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
    times = []
    for _ in range(n_bench):
        t0  = time.perf_counter()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return out, min(times)


def check(name, J_ref, J_test):
    denom   = jnp.max(jnp.abs(J_ref)) + 1e-12
    max_err = float(jnp.max(jnp.abs(J_ref - J_test)) / denom)
    status  = "PASS" if max_err < REL_TOL else "FAIL"
    print(f"  [{status}] max relative error = {max_err:.2e}  ({name})")
    assert max_err < REL_TOL, f"{name}: results diverge (err={max_err:.2e})"


# =========================================================
# 1. Steady-state (ray_trax_3D)
# =========================================================
print("\n=== Steady-state (ray_trax_3D) ===")

J1, t1 = timed_call(
    compute_radiation_field_from_source,
    emissivity, kappa, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
    use_sharding=False,
)
print(f"  1-GPU : {t1*1e3:.1f} ms")

J2, t2 = timed_call(
    compute_radiation_field_from_source,
    emissivity, kappa, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
    use_sharding=True,
)
print(f"  2-GPU : {t2*1e3:.1f} ms  (speedup {t1/t2:.2f}x)")
check("steady-state", J1, J2)


# =========================================================
# 2. Time-dependent (ray_trax_3D_tdep)
# =========================================================
print("\n=== Time-dependent (ray_trax_3D_tdep) ===")

box_diag  = float(jnp.sqrt(jnp.array(Nx**2 + Ny**2 + Nz**2)))
time_step = 1.2 * box_diag * STEP_SIZE   # enough to cross the full box

J3, t3 = timed_call(
    compute_radiation_field_from_source_with_time_step,
    emissivity, kappa, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE,
    time_step=time_step, radiation_velocity=1.0,
    use_sharding=False,
)
print(f"  1-GPU : {t3*1e3:.1f} ms")

J4, t4 = timed_call(
    compute_radiation_field_from_source_with_time_step,
    emissivity, kappa, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE,
    time_step=time_step, radiation_velocity=1.0,
    use_sharding=True,
)
print(f"  2-GPU : {t4*1e3:.1f} ms  (speedup {t3/t4:.2f}x)")
check("time-dependent", J3, J4)

print("\nAll tests passed.")
