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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cuda: needs a CUDA device (touches the cute backend)"
    )


def pytest_collection_modifyitems(config, items):
    # Only tests that reach the cute backend need a GPU; the reference-backend
    # wiring / mask / vmap tests run anywhere.
    if has_cuda():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------- kernel families
#
# Every cuda-marked test fans out over the vendored kernel families
# (sm80/sm90/sm100/sm120 parametrizations) and runs each one the device can
# actually execute, through the public selection mechanism: the fixture
# scopes fa4_jax.default_kernel_arch(...) around the test, so every call
# behaves as if the user passed kernel_arch=<family>. The rest appear as
# skips, so reports show exactly which implementations have coverage — the
# SM80-MMA kernels run fine on a B200, and several real varlen bugs lived
# only in that family. Codegen still targets the real device; only the
# kernel implementation is selected. The runnable/skip matrix is
# fa4_jax.attention.kernel_arch_unsupported — the same check the backend
# applies to user-supplied kernel_arch values.
#
# FA4_TEST_ARCHS=sm100,sm80 (or "80,100") restricts the sweep for quick runs.

KERNEL_ARCHS = (80, 90, 100, 120)


def _real_arch() -> int:
    cap = getattr(jax.devices()[0], "compute_capability", "0.0")
    major, _, minor = str(cap).partition(".")
    return int(major) * 10 + int(minor or 0)


def _selected_archs():
    spec = os.environ.get("FA4_TEST_ARCHS")
    if not spec:
        return KERNEL_ARCHS
    picked = tuple(int(a.strip().removeprefix("sm")) for a in spec.split(","))
    bad = [a for a in picked if a not in KERNEL_ARCHS]
    if bad:
        raise ValueError(f"FA4_TEST_ARCHS: unknown arch(s) {bad}; known: {KERNEL_ARCHS}")
    return picked


def pytest_generate_tests(metafunc):
    if "kernel_arch" in metafunc.fixturenames and metafunc.definition.get_closest_marker(
        "cuda"
    ):
        archs = _selected_archs()
        metafunc.parametrize(
            "kernel_arch", archs, ids=[f"sm{a}" for a in archs], indirect=True
        )


@pytest.fixture(autouse=True)
def kernel_arch(request):
    """Scope one kernel family around each cuda-marked test via the public
    ``fa4_jax.default_kernel_arch`` mechanism (explicit ``kernel_arch=``
    arguments in a test still win, as for any user). Unparametrized
    (non-cuda) tests get the native path untouched.
    """
    arch = getattr(request, "param", None)
    if arch is None or not has_cuda():
        yield None
        return
    from fa4_jax.attention import kernel_arch_unsupported

    reason = kernel_arch_unsupported(arch, _real_arch())
    if reason is not None:
        pytest.skip(f"kernel family sm{arch}: {reason}")
    import fa4_jax

    with fa4_jax.default_kernel_arch(arch):
        yield arch


TOL = {
    jnp.dtype(jnp.float32): dict(atol=2e-5, rtol=2e-5),
    # Flat tolerances, calibrated for the suite's unit-variance inputs (scale
    # 1/sqrt(d) -> logits with std ~1, a fairly flat softmax): measured kernel
    # error is ~8x below these. They are NOT valid for sharper attention,
    # where the bf16 rounding of P before the PV matmul dominates; those tests
    # use `fa_tol` instead.
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


# ---------------------------------------------------------------- FA-style tolerance
#
# flash-attention's test convention: the kernel's error against an fp32
# reference may be at most ~2x the error of the *same math done with half-
# precision rounding* against that fp32 reference. That makes the bound
# follow the actual conditioning of the inputs instead of a flat constant.


def half_rounded_attn(q, k, v, *, causal=False, window_size=None, scale=None, mask_mod=None):
    """The reference math with the kernel's roundings: P is rounded to the
    input dtype before the PV matmul, and the output to the input dtype."""
    import math

    from fa4_jax.reference import _expand_kv, dense_mask

    dtype = q.dtype
    f32 = jnp.float32
    b, s_q, h, d = q.shape
    s_kv = k.shape[1]
    scale = float(scale if scale is not None else 1.0 / math.sqrt(d))
    window = window_size if window_size is not None else (None, None)
    s = jnp.einsum("bqhd,bkhd->bhqk", q.astype(f32), _expand_kv(k, h).astype(f32)) * scale
    mask = dense_mask(b, h, s_q, s_kv, causal, window, mask_mod)
    if mask is not None:
        s = jnp.where(mask, s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1)
    if mask is not None:
        p = jnp.where(mask.any(-1, keepdims=True), p, 0.0)
    p = p.astype(dtype).astype(f32)
    out = jnp.einsum("bhqk,bkhd->bqhd", p, _expand_kv(v, h).astype(f32))
    return out.astype(dtype)


def fa_tol(want, half, dtype, *, mult=2.0):
    """Tolerance kwargs for assert_close: 2x the half-precision reference's
    own error plus the dtype's flat atol as a floor."""
    want = np.asarray(jnp.asarray(want, jnp.float32))
    half = np.asarray(jnp.asarray(half, jnp.float32))
    ref_err = float(np.abs(want - half).max())
    return dict(atol=mult * ref_err + TOL[jnp.dtype(dtype)]["atol"], rtol=0.0)
