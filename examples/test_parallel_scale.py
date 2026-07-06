"""
Meaningful multi-GPU speedup benchmark.
Uses autocvd to safely grab 2 free A100s.
Tests steady-state and time-dependent solvers at production-like scale.
"""
from autocvd import autocvd
autocvd(num_gpus=2)

import jax
import jax.numpy as jnp
import time

print(f"JAX      : {jax.__version__}")
print(f"Devices  : {jax.devices()}")
assert len(jax.devices()) == 2

from ray_trax.ray_trax_3D import compute_radiation_field_from_source
from ray_trax.ray_trax_3D_tdep import compute_radiation_field_from_source_with_time_step

# ---------------------------------------------------------------
# Grid + source (128^3, single point source at centre)
# ---------------------------------------------------------------
N         = 128
source_pos = jnp.array([N / 2.0, N / 2.0, N / 2.0])
kappa      = jnp.ones((N, N, N)) * 0.1
emissivity = jnp.zeros((N, N, N)).at[N // 2, N // 2, N // 2].set(1.0)

# ---------------------------------------------------------------
# Parameters chosen so each GPU is meaningfully loaded:
#   • num_rays = 8192  → 4096 rays per GPU
#   • ray_batch_size = 512  → peak 4 GB working set per batch per GPU
# ---------------------------------------------------------------
NUM_RAYS       = 8192
RAY_BATCH      = 512
MAX_STEPS      = 400
STEP_SIZE      = 0.5
REL_TOL        = 1e-4


def timed(fn, *a, n_warmup=1, n_bench=3, **kw):
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*a, **kw))
    ts = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a, **kw))
        ts.append(time.perf_counter() - t0)
    return fn(*a, **kw), min(ts)


def check(label, ref, test):
    denom   = float(jnp.max(jnp.abs(ref))) + 1e-12
    max_err = float(jnp.max(jnp.abs(ref - test))) / denom
    ok = max_err < REL_TOL
    print(f"    {'PASS' if ok else 'FAIL'}  max_rel_err = {max_err:.2e}  ({label})")
    assert ok, label


# ===============================================================
# 1. Steady-state  (ray_trax_3D)
# ===============================================================
print(f"\n=== Steady-state  N={N}  num_rays={NUM_RAYS}  max_steps={MAX_STEPS} ===")

J1, t1 = timed(
    compute_radiation_field_from_source,
    emissivity, kappa, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
    use_sharding=False, ray_batch_size=RAY_BATCH,
)
print(f"  1-GPU  {t1:.2f} s")

J2, t2 = timed(
    compute_radiation_field_from_source,
    emissivity, kappa, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE, max_steps=MAX_STEPS,
    use_sharding=True, ray_batch_size=RAY_BATCH,
)
print(f"  2-GPU  {t2:.2f} s   speedup {t1/t2:.2f}x")
check("steady-state", J1, J2)


# ===============================================================
# 2. Time-dependent  (ray_trax_3D_tdep)
# ===============================================================
print(f"\n=== Time-dependent  N={N}  num_rays={NUM_RAYS} ===")

box_diag  = float(jnp.sqrt(jnp.array(3 * N**2)))
time_step = 1.2 * box_diag * STEP_SIZE   # rays reach across the full box

J3, t3 = timed(
    compute_radiation_field_from_source_with_time_step,
    emissivity, kappa, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE,
    time_step=time_step, radiation_velocity=1.0,
    use_sharding=False, ray_batch_size=RAY_BATCH,
)
print(f"  1-GPU  {t3:.2f} s")

J4, t4 = timed(
    compute_radiation_field_from_source_with_time_step,
    emissivity, kappa, source_pos,
    num_rays=NUM_RAYS, step_size=STEP_SIZE,
    time_step=time_step, radiation_velocity=1.0,
    use_sharding=True, ray_batch_size=RAY_BATCH,
)
print(f"  2-GPU  {t4:.2f} s   speedup {t3/t4:.2f}x")
check("time-dependent", J3, J4)

print("\nAll benchmarks passed.")
