from __future__ import annotations

import os

# Share the GPU with whatever else is running: grow allocations on demand
# instead of preallocating, and cap the pool. Must be set before jax imports.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The reference computes in f32; on GPU that would silently be TF32 and the
# oracle would be noisier than the kernels it judges.
jax.config.update("jax_default_matmul_precision", "highest")


def has_cuda() -> bool:
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except RuntimeError:
        return False


def pytest_collection_modifyitems(config, items):
    # The whole suite exercises the cute backend (or compares against it).
    if has_cuda():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        item.add_marker(skip)


TOL = {
    jnp.dtype(jnp.float32): dict(atol=2e-5, rtol=2e-5),
    # Matching flash-attention's testing convention: kernel error is allowed a
    # small multiple of the reference implementation's own rounding error, and
    # half dtypes get flat tolerances scaled to their epsilon.
    jnp.dtype(jnp.bfloat16): dict(atol=3e-2, rtol=3e-2),
    jnp.dtype(jnp.float16): dict(atol=5e-3, rtol=5e-3),
}


def assert_close(got, want, *, dtype=None, atol=None, rtol=None, msg=""):
    got = np.asarray(jnp.asarray(got, jnp.float32))
    want = np.asarray(jnp.asarray(want, jnp.float32))
    tol = TOL[jnp.dtype(dtype)] if dtype is not None else dict(atol=atol, rtol=rtol)
    np.testing.assert_allclose(got, want, err_msg=msg, **tol)


def random_qkv(rng, b, s_q, s_kv, h_q, h_kv, d, d_v=None, dtype=jnp.float32):
    d_v = d if d_v is None else d_v
    mk = lambda *shape: jnp.asarray(rng.standard_normal(shape), dtype)
    return mk(b, s_q, h_q, d), mk(b, s_kv, h_kv, d), mk(b, s_kv, h_kv, d_v)
