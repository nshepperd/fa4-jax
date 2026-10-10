"""Kernel suite: the cute backend (vendored FA4 kernels + our JVP tangent
kernel) against the plain-jnp reference backend, for values, gradients and
JVPs.

Needs a CUDA device (sm_86/sm_120, sm_90 and sm_100 are what this repo targets).
Kernel compiles are cached per (head config, mask treedef), so the
parametrized sweeps mostly reuse a handful of compiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_close, fa_tol, half_rounded_attn, random_qkv

import fa4_jax
from fa4_jax import (
    FnMask,
    MaskMod,
    and_masks,
    causal_mask,
    document_mask,
    flash_attn,
    or_masks,
    sliding_window_mask,
)

pytestmark = pytest.mark.cuda

DTYPES = [jnp.bfloat16, jnp.float16]

# (b, s_q, s_kv, h_kv, gqa, d, d_v): tile-aligned, ragged tails, tiny, GQA,
# mixed head dims, s_q != s_kv both ways (bottom-right alignment), decode.
SHAPES = [
    (1, 128, 128, 1, 1, 64, None),
    (2, 256, 256, 4, 1, 64, None),
    (1, 200, 200, 2, 1, 64, None),
    (2, 37, 71, 2, 1, 64, None),
    (1, 1, 1, 1, 1, 64, None),
    (1, 1, 256, 2, 1, 64, None),  # decode: one query, many keys
    (1, 256, 64, 2, 1, 64, None),  # s_q > s_kv
    (2, 128, 128, 2, 2, 64, None),  # GQA
    (1, 192, 192, 1, 4, 64, None),  # GQA
    (1, 100, 164, 2, 2, 64, None),  # GQA + s_q != s_kv + ragged
    (1, 128, 128, 2, 1, 128, None),  # head_dim 128
    (1, 96, 96, 2, 1, 32, None),
    (1, 128, 128, 2, 1, 96, None),  # head_dim 96: padded to the 16-multiple tile
    (1, 128, 128, 2, 1, 64, 32),  # d_v != d
    (1, 128, 128, 2, 1, 64, 128),  # d_v > d
    (1, 128, 128, 2, 1, 16, None),  # smallest head dim (padded to the 32-multiple accumulators)
    (1, 128, 128, 2, 1, 48, None),  # 16 mod 32
    (2, 128, 128, 2, 2, 48, 16),  # GQA + mixed small head dims
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


def _tangents(q, k, v, seed):
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal(t.shape), q.dtype) for t in (q, k, v))


def _grads(q, k, v, cot=None, **kw):
    def loss(backend):
        def f(q, k, v):
            out = flash_attn(q, k, v, backend=backend, **kw)
            return out.sum() if cot is None else (out * cot).sum()

        return f

    got = jax.grad(loss("cute"), argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss("reference"), argnums=(0, 1, 2))(q, k, v)
    return got, want


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_fwd(shape, dtype, causal):
    q, k, v = _mk(shape, dtype)
    got, want = _both(q, k, v, causal=causal)
    assert_close(got, want, dtype=dtype)


WINDOWS = [(32, 32), (16, 0), (0, 16), (128, 7), (None, 5), (3, None)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("s_q,s_kv", [(200, 200), (80, 200), (200, 80)])
def test_fwd_window(dtype, window, s_q, s_kv):
    q, k, v = _mk((2, s_q, s_kv, 2, 1, 64, None), dtype)
    got, want = _both(q, k, v, window_size=window)
    assert_close(got, want, dtype=dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kw", [dict(), dict(causal=True), dict(window_size=(24, 8))])
def test_fwd_lse(shape, kw):
    """lse at 1e-3 is the sharpest edge detector here: one dropped key shifts
    it by ~1/s."""
    q, k, v = _mk(shape, jnp.bfloat16)
    got, lse_got = flash_attn(q, k, v, return_lse=True, backend="cute", **kw)
    want, lse_want = flash_attn(q, k, v, return_lse=True, backend="reference", **kw)
    assert_close(got, want, dtype=q.dtype)
    assert_close(lse_got, lse_want, atol=1e-3, rtol=1e-3)


def test_custom_scale():
    q, k, v = _mk((1, 128, 128, 2, 1, 64, None), jnp.bfloat16)
    got, want = _both(q, k, v, scale=0.5)
    assert_close(got, want, dtype=q.dtype)
    ref_default = flash_attn(q, k, v, backend="reference")
    assert np.abs(np.asarray(want - ref_default, np.float32)).max() > 1e-2


# ---------------------------------------------------------------- sharp softmax
#
# Unit-variance inputs with scale 1/sqrt(d) give a flat softmax, where the
# kernels' dominant rounding (P to bf16 before P@V) barely shows. Scaling q
# up makes rows peaky, moves the running max across KV blocks (online-softmax
# rescaling), and grows the error past the flat tolerances -- so these use
# the flash-attention rule: <= 2x the half-precision reference's own error.

SHARP_GAIN = 4.0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kw", [dict(), dict(causal=True), dict(window_size=(32, 16))])
def test_fwd_sharp(dtype, kw):
    q, k, v = _mk((2, 256, 256, 2, 1, 64, None), dtype)
    q = q * SHARP_GAIN
    got, want = _both(q, k, v, **kw)
    half = half_rounded_attn(q, k, v, **kw)
    assert_close(got, want, **fa_tol(want, half, dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kw", [dict(causal=True), dict(window_size=(32, 16))])
def test_grad_sharp(dtype, kw):
    q, k, v = _mk((2, 256, 256, 2, 1, 64, None), dtype)
    q = q * SHARP_GAIN
    got, want = _grads(q, k, v, **kw)
    half = jax.grad(lambda *a: half_rounded_attn(*a, **kw).sum(), argnums=(0, 1, 2))(q, k, v)
    for g, w, h, name in zip(got, want, half, "qkv"):
        assert_close(g, w, **fa_tol(w, h, dtype), msg=f"d{name}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_jvp_sharp(dtype):
    q, k, v = _mk((2, 256, 256, 2, 1, 64, None), dtype)
    q = q * SHARP_GAIN
    dq, dk, dv = _tangents(q, k, v, 31)
    f = lambda be: lambda q, k, v: flash_attn(q, k, v, causal=True, backend=be)
    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    _, t_half = jax.jvp(lambda *a: half_rounded_attn(*a, causal=True), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, **fa_tol(t_want, t_half, dtype))


# ---------------------------------------------------------------- mask mods

# Masks key the compile cache by class + static fields, so these could be
# built per call; FnMask is the exception (its lambda hashes by identity).
_WINDOW_MM = sliding_window_mask(24, 8)
_CAUSAL_MM = causal_mask()
_HEAD_MM = FnMask(lambda b, h, q, kv: (kv <= q) | (h == 0), label="h0-full")
_STRIDE_MM = FnMask(lambda b, h, q, kv: (kv <= q) & (kv % 3 == 0), label="stride3")


@dataclass(frozen=True)
class _PrefixMask(MaskMod):
    """A second array-leaf mask, to combine with document_mask (n_aux = 2)."""

    prefix_len: jax.Array

    def __call__(self, b, h, q, kv):
        return kv < self.prefix_len[b]


def _doc_ids(s, n_docs, b=2):
    return jnp.asarray(np.repeat(np.arange(n_docs), -(-s // n_docs))[:s])[None].repeat(b, 0).astype(jnp.int32)


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


def test_mask_mod_causal_is_top_left():
    """causal_mask() sees raw indices (top-left); causal=True is bottom-right.
    They differ when s_q != s_kv, and each backend agrees with itself."""
    q, k, v = _mk((1, 64, 128, 1, 1, 64, None), jnp.bfloat16)
    got, want = _both(q, k, v, mask_mod=_CAUSAL_MM)
    assert_close(got, want, dtype=q.dtype)
    native = flash_attn(q, k, v, causal=True, backend="cute")
    assert np.abs(np.asarray(got - native, np.float32)).max() > 0.1


def test_mask_mod_head_dependent():
    q, k, v = _mk((2, 128, 128, 3, 1, 64, None), jnp.bfloat16)
    got, want = _both(q, k, v, mask_mod=_HEAD_MM)
    assert_close(got, want, dtype=q.dtype)


def test_distinct_fnmasks_get_distinct_kernels():
    """Two FnMasks wrapping different lambdas must not share a compiled
    kernel (the fn is a static treedef field, hashed by identity)."""
    q, k, v = _mk((1, 128, 128, 1, 1, 64, None), jnp.bfloat16)
    a, want_a = _both(q, k, v, mask_mod=_HEAD_MM)
    b, want_b = _both(q, k, v, mask_mod=_STRIDE_MM)
    assert_close(a, want_a, dtype=q.dtype)
    assert_close(b, want_b, dtype=q.dtype)
    assert np.abs(np.asarray(a - b, np.float32)).max() > 0.1


@pytest.mark.parametrize("dtype", DTYPES)
def test_mask_mod_document_aux(dtype):
    s = 256
    q, k, v = _mk((2, s, s, 2, 1, 64, None), dtype)
    rng = np.random.default_rng(7)
    lens = rng.integers(16, 80, size=8)
    ids = np.repeat(np.arange(len(lens)), lens)[:s]
    ids = np.pad(ids, (0, s - len(ids)), constant_values=len(lens))
    docs = jnp.asarray(np.stack([ids, ids[::-1].copy()]), dtype=jnp.int32)
    got, want = _both(q, k, v, mask_mod=document_mask(docs))
    assert_close(got, want, dtype=dtype)


def test_mask_mod_combinators_and_multi_aux():
    """and_masks / or_masks on the kernels, including a mask with two array
    leaves (aux ordering through the treedef)."""
    s = 160
    q, k, v = _mk((2, s, s, 2, 1, 64, None), jnp.bfloat16)
    docs = _doc_ids(s, 3)
    prefix = _PrefixMask(jnp.asarray([40, 100], jnp.int32))
    for m in (
        and_masks(document_mask(docs), prefix),
        or_masks(document_mask(docs), prefix),
        and_masks(_CAUSAL_MM, or_masks(document_mask(docs), _WINDOW_MM)),
    ):
        got, want = _both(q, k, v, mask_mod=m)
        assert_close(got, want, dtype=q.dtype, msg=repr(m))
        got, want = _grads(q, k, v, mask_mod=m)
        for g, w, name in zip(got, want, "qkv"):
            assert_close(g, w, dtype=q.dtype, msg=f"{m!r} d{name}")


def test_mask_mod_aux_change_reuses_kernel():
    """Fresh mask instances with new array contents hit the compiled kernel."""
    import cutejax

    s = 128
    q, k, v = _mk((1, s, s, 1, 1, 64, None), jnp.bfloat16)
    ids1 = jnp.asarray(np.repeat([0, 1], [64, 64]))[None].astype(jnp.int32)
    ids2 = jnp.asarray(np.repeat([0, 1], [32, 96]))[None].astype(jnp.int32)
    o1 = flash_attn(q, k, v, mask_mod=document_mask(ids1), backend="cute")
    before = cutejax.cache_info()["size"]
    o2 = flash_attn(q, k, v, mask_mod=document_mask(ids2), backend="cute")
    assert cutejax.cache_info()["size"] == before, "aux data must not recompile"
    assert np.abs(np.asarray(o1 - o2, dtype=np.float32)).max() > 1e-3


# ---------------------------------------------------------------- gradients


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_grad(shape, dtype, causal):
    q, k, v = _mk(shape, dtype)
    rng = np.random.default_rng(1)
    cot = jnp.asarray(rng.standard_normal((*q.shape[:3], v.shape[3])), dtype)
    got, want = _grads(q, k, v, cot, causal=causal)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=dtype, msg=f"d{name}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window", [(32, 32), (16, 0), (0, 16), (None, 5)])
@pytest.mark.parametrize("s_q,s_kv", [(256, 256), (100, 200), (200, 100)])
def test_grad_window(dtype, window, s_q, s_kv):
    q, k, v = _mk((1, s_q, s_kv, 2, 1, 64, None), dtype)
    got, want = _grads(q, k, v, window_size=window)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=dtype, msg=f"d{name}")


@pytest.mark.parametrize("shape", [(2, 200, 200, 2, 4, 64, None), (1, 128, 128, 1, 2, 64, 32), (1, 64, 128, 1, 8, 64, None)])
@pytest.mark.parametrize("kw", [dict(window_size=(32, 8)), dict(mask_mod=_WINDOW_MM), dict(causal=True)])
def test_grad_gqa(shape, kw):
    """GQA backward: dK/dV accumulate in fp32 across the q heads of a kv head,
    then go through the postprocess (dK scaled, dV not). Ragged tails, d_v !=
    d, s_q != s_kv, ratio 8, and every masking path, since the accumulator
    layouts are padded."""
    q, k, v = _mk(shape, jnp.bfloat16)
    rng = np.random.default_rng(1)
    cot = jnp.asarray(rng.standard_normal((*q.shape[:3], v.shape[3])), jnp.bfloat16)
    got, want = _grads(q, k, v, cot, **kw)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=q.dtype, msg=f"d{name}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_grad_mask_mod(dtype):
    """mask_mod in reverse mode (local addition to the vendored backward)."""
    q, k, v = _mk((2, 160, 160, 2, 1, 64, None), dtype)
    got, want = _grads(q, k, v, mask_mod=_WINDOW_MM)
    for g, w, name in zip(got, want, "qkv"):
        assert_close(g, w, dtype=dtype, msg=f"d{name}")


def test_grad_mask_mod_aux():
    s = 192
    q, k, v = _mk((2, s, s, 2, 1, 64, None), jnp.bfloat16)
    got, want = _grads(q, k, v, mask_mod=document_mask(_doc_ids(s, 4)))
    for g, w in zip(got, want):
        assert_close(g, w, dtype=q.dtype)


def test_grad_with_unused_lse():
    """return_lse=True, differentiating out only: the lse cotangent is a
    symbolic zero, so the cute backward (which has no d_lse) is used."""
    q, k, v = _mk((1, 128, 128, 2, 1, 64, None), jnp.bfloat16)
    loss = lambda be: lambda q, k, v: flash_attn(q, k, v, causal=True, return_lse=True, backend=be)[0].sum()
    got = jax.grad(loss("cute"), argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss("reference"), argnums=(0, 1, 2))(q, k, v)
    for g, w in zip(got, want):
        assert_close(g, w, dtype=q.dtype)


def test_grad_through_lse_unsupported():
    q, k, v = _mk((1, 64, 64, 1, 1, 64, None), jnp.bfloat16)
    f = lambda q: flash_attn(q, k, v, return_lse=True, backend="cute")[1].sum()
    with pytest.raises(NotImplementedError, match="lse"):
        jax.grad(f)(q)


def test_grad_partial_argnums():
    """grad w.r.t. k only: q/v tangents are symbolic zeros into the JVP rule."""
    q, k, v = _mk((1, 128, 128, 2, 1, 64, None), jnp.bfloat16)
    loss = lambda be: lambda k: flash_attn(q, k, v, causal=True, backend=be).sum()
    assert_close(jax.grad(loss("cute"))(k), jax.grad(loss("reference"))(k), dtype=k.dtype)


def test_grad_empty_rows_finite():
    """Rows with no visible key: zero output, -inf lse, and the backward must
    stay finite (P = 0 on those rows, not NaN from exp(-inf - -inf))."""
    q, k, v = _mk((1, 64, 64, 1, 1, 64, None), jnp.bfloat16)
    got, want = _grads(q, k, v, window_size=(None, -3))
    for g, w, name in zip(got, want, "qkv"):
        assert np.isfinite(np.asarray(g, np.float32)).all(), name
        assert_close(g, w, dtype=q.dtype, msg=f"d{name}")
    assert (np.asarray(got[0][:, :3], np.float32) == 0).all()


@pytest.mark.parametrize("native", [dict(causal=True), dict(window_size=(48, 16))])
def test_native_mask_combines_with_mask_mod(native):
    """causal/window + mask_mod fold into one predicate on the kernels
    (regression: the kernels' AttentionMask would otherwise silently drop the
    mod). Checked for fwd, grad, and jvp."""
    s = 160
    q, k, v = _mk((2, s, s, 2, 1, 64, None), jnp.bfloat16)
    kw = dict(mask_mod=document_mask(_doc_ids(s, 4)), **native)

    got, want = _both(q, k, v, **kw)
    assert_close(got, want, dtype=q.dtype, msg="fwd")

    gg, gw = _grads(q, k, v, **kw)
    for g, w in zip(gg, gw):
        assert_close(g, w, dtype=q.dtype, msg="grad")

    dq, dk, dv = _tangents(q, k, v, 9)
    f = lambda backend: lambda q, k, v: flash_attn(q, k, v, backend=backend, **kw)
    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, dtype=q.dtype, msg="jvp tangent")


# ---------------------------------------------------------------- jvp


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_jvp(shape, dtype, causal):
    q, k, v = _mk(shape, dtype)
    dq, dk, dv = _tangents(q, k, v, 2)

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, causal=causal, backend=backend)

    o_got, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    o_want, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(o_got, o_want, dtype=dtype, msg="primal")
    assert_close(t_got, t_want, dtype=dtype, msg="tangent")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("window", [(32, 32), (16, 0)])
@pytest.mark.parametrize("s_q,s_kv", [(200, 200), (80, 200), (200, 80)])
def test_jvp_window(dtype, window, s_q, s_kv):
    q, k, v = _mk((2, s_q, s_kv, 2, 1, 64, None), dtype)
    dq, dk, dv = _tangents(q, k, v, 3)

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, window_size=window, backend=backend)

    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, dtype=dtype)


def test_jvp_mask_mod_and_aux():
    """Unlike reverse mode, forward mode supports mask mods on the kernels."""
    s = 160
    q, k, v = _mk((2, s, s, 2, 1, 64, None), jnp.bfloat16)
    dq, dk, dv = _tangents(q, k, v, 4)
    ids = _doc_ids(s, 4)

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, mask_mod=document_mask(ids), backend=backend)

    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, dtype=q.dtype)


def test_jvp_aux_tangent_rejected():
    """A float mask leaf with a tangent is rejected before any kernel runs."""
    q, k, v = _mk((1, 64, 64, 1, 1, 64, None), jnp.bfloat16)
    ids = jnp.zeros((1, 64), jnp.float32)
    f = lambda ids: flash_attn(q, k, v, mask_mod=document_mask(ids), backend="cute")
    with pytest.raises(NotImplementedError, match="aux"):
        jax.jvp(f, (ids,), (jnp.ones_like(ids),))


def test_jvp_lse_tangent():
    """t_lse = mu comes out of the tangent kernel's second output."""
    q, k, v = _mk((2, 128, 128, 2, 1, 64, None), jnp.bfloat16)
    dq, dk, dv = _tangents(q, k, v, 5)

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, return_lse=True, backend=backend)[1]

    _, m_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, m_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(m_got, m_want, atol=5e-3, rtol=5e-3)


def test_jvp_gqa():
    """Forward mode supports GQA (nothing accumulates into kv shapes)."""
    q, k, v = _mk((2, 128, 128, 2, 2, 64, None), jnp.bfloat16)
    dq, dk, dv = _tangents(q, k, v, 6)

    def f(backend):
        return lambda q, k, v: flash_attn(q, k, v, causal=True, backend=backend)

    _, t_got = jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))
    _, t_want = jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))
    assert_close(t_got, t_want, dtype=q.dtype)


def _poison_shared_memory():
    """Run XLA reductions over NaN / -inf so the SMs' shared memory is left
    full of non-finite values (what an earlier masked-softmax leaves behind)."""
    x = jnp.full((2048, 2048), -jnp.inf, jnp.float32)

    @jax.jit
    def f(x):
        return (
            jnp.max(x, axis=1).sum()
            + jax.nn.softmax(x, axis=-1).sum()
            + (x.astype(jnp.bfloat16) @ x.astype(jnp.bfloat16)).sum()
        )

    f(x).block_until_ready()


@pytest.mark.parametrize("kw", [dict(), dict(causal=True), dict(window_size=(16, 8))])
@pytest.mark.parametrize("shape", [(2, 37, 71, 2, 1, 64, None), (1, 1, 200, 1, 1, 64, None)])
def test_ragged_tail_immune_to_stale_smem(shape, kw):
    """Regression: the KV tiles' seqlen tail is never loaded from gmem, so
    whatever the previous kernel left in shared memory sits there. The
    forward only needs the *scores* masked (P = 0 there), but the tangent
    kernel also forms tS from those columns and 0 * (stale inf) = NaN poisoned
    every row. The tail must be zero-filled (or otherwise excluded) in all
    three passes."""
    q, k, v = _mk(shape, jnp.bfloat16)
    dq, dk, dv = _tangents(q, k, v, 40)
    f = lambda be: lambda q, k, v: flash_attn(q, k, v, backend=be, **kw)
    jit_fwd = jax.jit(f("cute"))
    jit_jvp = jax.jit(lambda q, k, v, dq, dk, dv: jax.jvp(f("cute"), (q, k, v), (dq, dk, dv))[1])
    jit_grad = jax.jit(jax.grad(lambda q, k, v: f("cute")(q, k, v).sum(), argnums=(0, 1, 2)))
    for op, args, want in (
        (jit_fwd, (q, k, v), f("reference")(q, k, v)),
        (jit_jvp, (q, k, v, dq, dk, dv), jax.jvp(f("reference"), (q, k, v), (dq, dk, dv))[1]),
    ):
        op(*args).block_until_ready()  # compile first, then poison, then run
        _poison_shared_memory()
        got = op(*args)
        assert not np.isnan(np.asarray(got, np.float32)).any()
        assert_close(got, want, dtype=q.dtype)
    jit_grad(q, k, v)
    _poison_shared_memory()
    got = jit_grad(q, k, v)
    want = jax.grad(lambda q, k, v: f("reference")(q, k, v).sum(), argnums=(0, 1, 2))(q, k, v)
    for g, w, name in zip(got, want, "qkv"):
        assert not np.isnan(np.asarray(g, np.float32)).any(), name
        assert_close(g, w, dtype=q.dtype, msg=f"d{name}")


def test_empty_rows_zero_not_nan():
    """Fully-masked rows: forward zeros and tangent zeros, no NaNs."""
    q, k, v = _mk((1, 64, 64, 1, 1, 64, None), jnp.bfloat16)
    dq, dk, dv = _tangents(q, k, v, 8)
    f = lambda q, k, v: flash_attn(q, k, v, window_size=(-8, 16), backend="cute")
    out, t_out = jax.jvp(f, (q, k, v), (dq, dk, dv))
    out, t_out = np.asarray(out, np.float32), np.asarray(t_out, np.float32)
    assert not np.isnan(out).any() and not np.isnan(t_out).any()
    assert (out[0, -8:] == 0).all() and (t_out[0, -8:] == 0).all()


# ---------------------------------------------------------------- jit / lowering


def test_jit_fwd_grad_jvp():
    """Under an outer jax.jit the primitives go through their mlir lowering
    (lower_fun) rather than the eager impl; numbers must be identical."""
    q, k, v = _mk((1, 128, 128, 2, 1, 64, None), jnp.bfloat16)
    dq, dk, dv = _tangents(q, k, v, 10)
    f = lambda q, k, v: flash_attn(q, k, v, causal=True, return_lse=True, backend="cute")
    loss = lambda q, k, v: f(q, k, v)[0].sum()
    tan = lambda q, k, v, dq, dk, dv: jax.jvp(f, (q, k, v), (dq, dk, dv))[1]

    for eager, jitted in (
        (f, jax.jit(f)),
        (jax.grad(loss, argnums=(0, 1, 2)), jax.jit(jax.grad(loss, argnums=(0, 1, 2)))),
    ):
        for a, b in zip(eager(q, k, v), jitted(q, k, v)):
            assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(tan(q, k, v, dq, dk, dv), jax.jit(tan)(q, k, v, dq, dk, dv)):
        assert_close(a, b, atol=0, rtol=0)


def test_remat():
    q, k, v = _mk((1, 128, 128, 2, 1, 64, None), jnp.bfloat16)
    loss = lambda q, k, v: flash_attn(q, k, v, causal=True, backend="cute").sum()
    got = jax.grad(jax.checkpoint(loss), argnums=(0, 1, 2))(q, k, v)
    want = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    for a, b in zip(got, want):
        assert_close(a, b, atol=0, rtol=0)


def test_public_surface():
    """fa4_jax package exposes what the README promises."""
    assert {"flash_attn", "MaskMod", "reference_attn"} <= set(fa4_jax.__all__)


# ---------------------------------------------------------------- head-dim envelope
#
# Whether a large / mixed head-dim pair fits is family- AND device-dependent:
# the SM80-MMA tiles fit hd 256 in sm_80's 163 KB but not in a GeForce's
# 99 KB, and the tcgen05 kernels stop at 128. The contract is that every
# pair either raises the clean "cannot run head_dim" ValueError up front or
# computes correctly -- never a launch that fails inside CUDA and poisons the
# context (which would take every later test down with it). Both fwd and bwd
# have their own SMEM budgets, so both are swept. The SM80-MMA families
# (sm80, sm120) must actually run all of them, even in 99 KB: the forward via
# a 64x64 tile, the backward via the smaller-tile rows of _bwd_cfgs_sm80.
# s_kv = 150 is a multiple of no kv tile, so the seqlen mask runs on partial
# tiles.

LARGE_HEAD_DIMS = [
    (192, 192),  # bwd: 64x32, 8 warps (2x4 dK/dV warp grid)
    (224, 224),  # bwd: 32x32 -- the 64-wide rows don't tile 224
    (256, 256),  # bwd: 64x32, 8 warps, V in registers
    (128, 256),  # d_v > d
    (256, 128),  # d > d_v
    (256, 64),  # bwd: 64x64 still fits
    (64, 192),
]


def _fits_or_rejects(fn, *args):
    try:
        return fn(*args)
    except ValueError as e:
        assert "cannot run head_dim=" in str(e), e
        return None


@pytest.mark.parametrize("d,d_v", LARGE_HEAD_DIMS, ids=lambda x: str(x))
@pytest.mark.parametrize("causal", [False, True])
def test_large_head_dims(d, d_v, causal, kernel_arch):
    run = (lambda fn: fn()) if kernel_arch in (80, 120) else _fits_or_rejects
    q, k, v = _mk((1, 96, 150, 2, 1, d, d_v), jnp.bfloat16)
    got = run(lambda: flash_attn(q, k, v, backend="cute", causal=causal))
    if got is not None:
        want = flash_attn(q, k, v, backend="reference", causal=causal)
        assert_close(got, want, dtype=q.dtype)

    rng = np.random.default_rng(1)
    cot = jnp.asarray(rng.standard_normal((*q.shape[:3], d_v)), q.dtype)
    grads = run(lambda: _grads(q, k, v, cot, causal=causal))
    if grads is not None:
        got, want = grads
        for g, w, name in zip(got, want, "qkv"):
            assert_close(g, w, dtype=q.dtype, msg=f"d{name}")


def test_kernel_arch_validation():
    """Unrunnable kernel_arch values fail up front with a clear message,
    not deep inside kernel construction. The wgmma (sm_90) and tcgen05
    (sm_100) families each run on exactly one generation, so whichever the
    device is not gives an unrunnable-but-known family."""
    from conftest import _real_arch

    import fa4_jax

    q, k, v = _mk((1, 64, 64, 1, 1, 64, None), jnp.bfloat16)
    foreign = 100 if _real_arch() // 10 == 9 else 90
    with pytest.raises(ValueError, match="need"):
        flash_attn(q, k, v, backend="cute", kernel_arch=foreign)
    with pytest.raises(ValueError, match="unknown kernel family"):
        flash_attn(q, k, v, backend="cute", kernel_arch=75)

    with fa4_jax.default_kernel_arch(foreign), pytest.raises(ValueError, match="need"):
        flash_attn(q, k, v, backend="cute")


def test_kernel_arch_param_overrides_fixture():
    """The public kernel_arch argument selects the family per call,
    independent of the device (and of this suite's family fixture): both
    explicit choices must agree with the reference and with each other."""
    q, k, v = _mk((1, 200, 200, 2, 1, 64, None), jnp.bfloat16)
    want = flash_attn(q, k, v, causal=True, backend="reference")
    outs = {}
    for arch in (80, 120):
        outs[arch] = flash_attn(q, k, v, causal=True, backend="cute", kernel_arch=arch)
        assert_close(outs[arch], want, dtype=q.dtype, msg=f"kernel_arch={arch}")
    grad80 = jax.grad(
        lambda q: flash_attn(q, k, v, causal=True, backend="cute", kernel_arch=80).sum()
    )(q)
    grad_ref = jax.grad(
        lambda q: flash_attn(q, k, v, causal=True, backend="reference").sum()
    )(q)
    assert_close(grad80, grad_ref, dtype=q.dtype)
