"""GPU suite: the cute backend (vendored FA4 kernels + our JVP tangent
kernel) against the reference backend, which the CPU suite has already pinned
to jax.nn.dot_product_attention and plain-JAX autodiff.

Run with `pytest -m gpu` on a machine with a CUDA device (sm_80+ family;
sm_86/sm_120 are what this repo targets). Kernel compiles are cached per
(head config, mask object), so the parametrized sweeps mostly reuse a handful
of compiles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_close, random_qkv

import fa4_jax
from fa4_jax import (
    MaskMod,
    causal_mask,
    document_mask,
    flash_attn,
    sliding_window_mask,
)

pytestmark = pytest.mark.gpu

DTYPES = [jnp.bfloat16, jnp.float16]

# (b, s_q, s_kv, h_kv, gqa, d, d_v): tile-aligned, ragged tails, tiny, GQA,
# mixed head dims.
SHAPES = [
    (1, 128, 128, 1, 1, 64, None),
    (2, 256, 256, 4, 1, 64, None),
    (1, 200, 200, 2, 1, 64, None),
    (2, 37, 71, 2, 1, 64, None),
    (1, 1, 1, 1, 1, 64, None),
    (2, 128, 128, 2, 2, 64, None),  # GQA
    (1, 192, 192, 1, 4, 64, None),  # GQA
    (1, 128, 128, 2, 1, 128, None),  # head_dim 128
    (1, 96, 96, 2, 1, 32, None),
    (1, 128, 128, 2, 1, 64, 32),  # d_v != d
]


def _mk(shape, dtype, seed=0):
    b, s_q, s_kv, h_kv, gqa, d, d_v = shape
    rng = np.random.default_rng(seed)
    q, k, v = random_qkv(rng, b, s_q, s_kv, h_kv * gqa, h_kv, d, d_v, dtype=dtype)
    return q, k, v


def _both(q, k, v, **kw):
    got = flash_attn(q, k, v, backend="cute", **kw)
    want = flash_attn(q, k, v, backend="reference", **kw)
    return got, want


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_fwd(shape, dtype, causal):
    if causal and shape[1] != shape[2]:
        pytest.skip("causal alignment for s_q != s_kv not pinned down")
    q, k, v = _mk(shape, dtype)
    got, want = _both(q, k, v, causal=causal)
    assert_close(got, want, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window", [(32, 32), (16, 0), (0, 16), (128, 7), (None, 5), (3, None)])
def test_fwd_window(dtype, window):
    q, k, v = _mk((2, 200, 200, 2, 1, 64, None), dtype)
    got, want = _both(q, k, v, window_size=window)
    assert_close(got, want, dtype=dtype)


@pytest.mark.parametrize("shape", SHAPES)
def test_fwd_lse(shape):
    q, k, v = _mk(shape, jnp.bfloat16)
    got, lse_got = flash_attn(q, k, v, return_lse=True, backend="cute")
    want, lse_want = flash_attn(q, k, v, return_lse=True, backend="reference")
    assert_close(got, want, dtype=q.dtype)
    assert_close(lse_got, lse_want, atol=1e-3, rtol=1e-3)


# one MaskMod instance per test session: each is its own compile-cache key
_WINDOW_MM = sliding_window_mask(24, 8)
_CAUSAL_MM = causal_mask()
_DOC_MM = document_mask()
_HEAD_MM = MaskMod(lambda b, h, q, kv, aux: (kv <= q) | (h == 0), name="h0-full")


@pytest.mark.parametrize("dtype", DTYPES)
def test_mask_mod_window(dtype):
    q, k, v = _mk((2, 160, 160, 2, 1, 64, None), dtype)
    got, want = _both(q, k, v, mask_mod=_WINDOW_MM)
    assert_close(got, want, dtype=dtype)


def test_mask_mod_equals_native_causal():
    q, k, v = _mk((2, 128, 128, 2, 1, 64, None), jnp.bfloat16)
    got = flash_attn(q, k, v, mask_mod=_CAUSAL_MM, backend="cute")
    want = flash_attn(q, k, v, causal=True, backend="cute")
    assert_close(got, want, dtype=q.dtype)


def test_mask_mod_head_dependent():
    q, k, v = _mk((2, 128, 128, 3, 1, 64, None), jnp.bfloat16)
    got, want = _both(q, k, v, mask_mod=_HEAD_MM)
    assert_close(got, want, dtype=q.dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mask_mod_document_aux(dtype):
    s = 256
    q, k, v = _mk((2, s, s, 2, 1, 64, None), dtype)
    rng = np.random.default_rng(7)
    lens = rng.integers(16, 80, size=8)
    ids = np.repeat(np.arange(len(lens)), lens)[:s]
    ids = np.pad(ids, (0, s - len(ids)), constant_values=len(lens))
    docs = jnp.asarray(np.stack([ids, ids[::-1].copy()]), dtype=jnp.int32)
    got, want = _both(q, k, v, mask_mod=_DOC_MM, aux_tensors=(docs,))
    assert_close(got, want, dtype=dtype)


def test_mask_mod_aux_change_reuses_kernel():
    import cutejax

    s = 128
    q, k, v = _mk((1, s, s, 1, 1, 64, None), jnp.bfloat16)
    ids1 = jnp.asarray(np.repeat([0, 1], [64, 64]))[None].astype(jnp.int32)
    ids2 = jnp.asarray(np.repeat([0, 1], [32, 96]))[None].astype(jnp.int32)
    o1 = flash_attn(q, k, v, mask_mod=_DOC_MM, aux_tensors=(ids1,), backend="cute")
    before = cutejax.cache_info()["size"]
    o2 = flash_attn(q, k, v, mask_mod=_DOC_MM, aux_tensors=(ids2,), backend="cute")
    assert cutejax.cache_info()["size"] == before, "aux data must not recompile"
    assert np.abs(np.asarray(o1 - o2, dtype=np.float32)).max() > 1e-3


# ---------------------------------------------------------------- gradients


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape", [s for s in SHAPES if s[3] * s[4] == s[3]]  # no GQA (bwd restriction)
)
@pytest.mark.parametrize("causal", [False, True])
def test_grad(shape, dtype, causal):
    if causal and shape[1] != shape[2]:
        pytest.skip("causal alignment for s_q != s_kv not pinned down")
    q, k, v = _mk(shape, dtype)
    rng = np.random.default_rng(1)
    cot = jnp.asarray(rng.standard_normal((*q.shape[:3], v.shape[3])), dtype)

    def loss(backend):
        return lambda q, k, v: (flash_attn(q, k, v, causal=causal, backend=backend) * cot).sum()

    got = jax.grad(loss("cute"), argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss("reference"), argnums=(0, 1, 2))(q, k, v)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=dtype, msg=f"d{name}")


@pytest.mark.parametrize("window", [(32, 32), (16, 0)])
def test_grad_window(window):
    q, k, v = _mk((1, 256, 256, 2, 1, 64, None), jnp.bfloat16)

    def loss(backend):
        return lambda q, k, v: flash_attn(q, k, v, window_size=window, backend=backend).sum()

    got = jax.grad(loss("cute"), argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss("reference"), argnums=(0, 1, 2))(q, k, v)
    for g, w in zip(got, want):
        assert_close(g, w, dtype=q.dtype)


def test_grad_gqa_raises():
    q, k, v = _mk((1, 64, 64, 2, 2, 64, None), jnp.bfloat16)
    with pytest.raises(NotImplementedError, match="GQA"):
        jax.grad(lambda q: flash_attn(q, k, v, backend="cute").sum())(q)


@pytest.mark.parametrize("dtype", DTYPES)
def test_grad_mask_mod(dtype):
    """mask_mod in reverse mode (local addition to the vendored backward)."""
    q, k, v = _mk((2, 160, 160, 2, 1, 64, None), dtype)

    def loss(backend):
        return lambda q, k, v: flash_attn(q, k, v, mask_mod=_WINDOW_MM, backend=backend).sum()

    got = jax.grad(loss("cute"), argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss("reference"), argnums=(0, 1, 2))(q, k, v)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=dtype, msg=f"d{name}")


def test_grad_mask_mod_aux():
    s = 192
    q, k, v = _mk((2, s, s, 2, 1, 64, None), jnp.bfloat16)
    ids = jnp.asarray(np.repeat(np.arange(4), s // 4))[None].repeat(2, 0).astype(jnp.int32)

    def loss(backend):
        return lambda q, k, v: flash_attn(
            q, k, v, mask_mod=_DOC_MM, aux_tensors=(ids,), backend=backend
        ).sum()

    got = jax.grad(loss("cute"), argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss("reference"), argnums=(0, 1, 2))(q, k, v)
    for g, w in zip(got, want):
        assert_close(g, w, dtype=q.dtype)


@pytest.mark.parametrize("native", [dict(causal=True), dict(window_size=(48, 16))])
def test_native_mask_combines_with_mask_mod(native):
    """causal/window + mask_mod fold into one predicate on the kernels
    (regression: the kernels' AttentionMask would otherwise silently drop the
    mod). Checked for fwd, grad, and jvp."""
    s = 160
    q, k, v = _mk((2, s, s, 2, 1, 64, None), jnp.bfloat16)
    ids = jnp.asarray(np.repeat(np.arange(4), s // 4))[None].repeat(2, 0).astype(jnp.int32)
    kw = dict(mask_mod=_DOC_MM, aux_tensors=(ids,), **native)

    got, want = _both(q, k, v, **kw)
    assert_close(got, want, dtype=q.dtype, msg="fwd")

    def loss(backend):
        return lambda q, k, v: flash_attn(q, k, v, backend=backend, **kw).sum()

    gg = jax.grad(loss("cute"), argnums=(0, 1, 2))(q, k, v)
    gw = jax.grad(loss("reference"), argnums=(0, 1, 2))(q, k, v)
    for g, w in zip(gg, gw):
        assert_close(g, w, dtype=q.dtype, msg="grad")

    rng = np.random.default_rng(9)
    dq, dk, dv = (jnp.asarray(rng.standard_normal(t.shape), q.dtype) for t in (q, k, v))
    f = lambda backend: lambda q, k, v: flash_attn(q, k, v, backend=backend, **kw)
    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, dtype=q.dtype, msg="jvp tangent")


# ---------------------------------------------------------------- jvp


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_jvp(shape, dtype, causal):
    if causal and shape[1] != shape[2]:
        pytest.skip("causal alignment for s_q != s_kv not pinned down")
    q, k, v = _mk(shape, dtype)
    rng = np.random.default_rng(2)
    dq, dk, dv = (jnp.asarray(rng.standard_normal(t.shape), dtype) for t in (q, k, v))

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, causal=causal, backend=backend)

    o_got, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    o_want, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(o_got, o_want, dtype=dtype, msg="primal")
    assert_close(t_got, t_want, dtype=dtype, msg="tangent")


@pytest.mark.parametrize("window", [(32, 32), (16, 0)])
def test_jvp_window(window):
    q, k, v = _mk((2, 200, 200, 2, 1, 64, None), jnp.bfloat16)
    rng = np.random.default_rng(3)
    dq, dk, dv = (jnp.asarray(rng.standard_normal(t.shape), q.dtype) for t in (q, k, v))

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, window_size=window, backend=backend)

    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, dtype=q.dtype)


def test_jvp_mask_mod_and_aux():
    """Unlike reverse mode, forward mode supports mask mods on the kernels."""
    s = 160
    q, k, v = _mk((2, s, s, 2, 1, 64, None), jnp.bfloat16)
    rng = np.random.default_rng(4)
    dq, dk, dv = (jnp.asarray(rng.standard_normal(t.shape), q.dtype) for t in (q, k, v))
    ids = jnp.asarray(np.repeat(np.arange(4), s // 4))[None].repeat(2, 0).astype(jnp.int32)

    def f(backend):
        return lambda q, k, v: flash_attn(
            q, k, v, mask_mod=_DOC_MM, aux_tensors=(ids,), backend=backend
        )

    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, dtype=q.dtype)


def test_jvp_lse_tangent():
    """t_lse = mu comes out of the tangent kernel's second output."""
    q, k, v = _mk((2, 128, 128, 2, 1, 64, None), jnp.bfloat16)
    rng = np.random.default_rng(5)
    dq, dk, dv = (jnp.asarray(rng.standard_normal(t.shape), q.dtype) for t in (q, k, v))

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, return_lse=True, backend=backend)[1]

    _, m_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, m_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(m_got, m_want, atol=5e-3, rtol=5e-3)


def test_jvp_gqa():
    """Forward mode supports GQA (nothing accumulates into kv shapes)."""
    q, k, v = _mk((2, 128, 128, 2, 2, 64, None), jnp.bfloat16)
    rng = np.random.default_rng(6)
    dq, dk, dv = (jnp.asarray(rng.standard_normal(t.shape), q.dtype) for t in (q, k, v))

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, causal=True, backend=backend)

    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, dtype=q.dtype)


def test_empty_rows_zero_not_nan():
    """Fully-masked rows: forward zeros and tangent zeros, no NaNs."""
    q, k, v = _mk((1, 64, 64, 1, 1, 64, None), jnp.bfloat16)
    rng = np.random.default_rng(8)
    dq, dk, dv = (jnp.asarray(rng.standard_normal(t.shape), q.dtype) for t in (q, k, v))
    f = lambda q, k, v: flash_attn(q, k, v, window_size=(-8, 16), backend="cute")
    out, t_out = jax.jvp(f, (q, k, v), (dq, dk, dv))
    out, t_out = np.asarray(out, np.float32), np.asarray(t_out, np.float32)
    assert not np.isnan(out).any() and not np.isnan(t_out).any()
    assert (out[0, -8:] == 0).all() and (t_out[0, -8:] == 0).all()


def test_reference_vs_fa4_module_docstring_layout():
    """fa4_jax package exposes what the README promises."""
    assert {"flash_attn", "MaskMod", "reference_attn"} <= set(fa4_jax.__all__)
