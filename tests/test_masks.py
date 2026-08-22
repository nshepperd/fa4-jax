"""MaskMod semantics: the dense JAX evaluation path, the combinators, and
equivalence between native kernel options (causal/window) and their mask-mod
spellings — which is exactly the property that lets the GPU suite trust the
mask-mod path."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from conftest import assert_close, random_qkv
from hypothesis import given
from hypothesis import strategies as st

from fa4_jax import (
    MaskMod,
    causal_mask,
    document_mask,
    flash_attn,
    reference_attn,
    sliding_window_mask,
)
from fa4_jax.masks import and_masks, or_masks
from fa4_jax.reference import dense_mask


def _dense(mask, b=2, h=3, s_q=7, s_kv=9, aux=()):
    return np.asarray(dense_mask(b, h, s_q, s_kv, False, (None, None), mask, aux))


def test_causal_mask_dense():
    m = _dense(causal_mask())
    q, kv = np.mgrid[0:7, 0:9]
    np.testing.assert_array_equal(m, np.broadcast_to(kv <= q, (2, 3, 7, 9)))


@given(left=st.one_of(st.none(), st.integers(-3, 12)), right=st.one_of(st.none(), st.integers(-3, 12)))
def test_sliding_window_mask_dense(left, right):
    if left is None and right is None:
        return
    m = _dense(sliding_window_mask(left, right))
    q, kv = np.mgrid[0:7, 0:9]
    want = np.ones_like(q, dtype=bool)
    if left is not None:
        want &= q - kv <= left
    if right is not None:
        want &= kv - q <= right
    np.testing.assert_array_equal(m, np.broadcast_to(want, (2, 3, 7, 9)))


def test_document_mask_dense():
    ids = jnp.asarray([[0, 0, 1, 1, 1, 2, 2, 2, 2]])
    m = _dense(document_mask(), b=1, s_q=9, s_kv=9, aux=(ids,))
    want = np.asarray(ids[0])[:, None] == np.asarray(ids[0])[None, :]
    np.testing.assert_array_equal(m, np.broadcast_to(want, (1, 3, 9, 9)))


def test_combinators():
    a, b = causal_mask(), sliding_window_mask(2, None)
    m_and = _dense(and_masks(a, b))
    m_or = _dense(or_masks(a, b))
    np.testing.assert_array_equal(m_and, _dense(a) & _dense(b))
    np.testing.assert_array_equal(m_or, _dense(a) | _dense(b))


def test_head_dependent_mask():
    """The predicate sees the head index (flex-attention alibi-style usage)."""
    m = MaskMod(lambda b, h, q, kv, aux: (kv <= q) | (h == 0), name="h0-full")
    d = _dense(m)
    assert d[:, 0].all()
    q, kv = np.mgrid[0:7, 0:9]
    np.testing.assert_array_equal(d[:, 1:], np.broadcast_to(kv <= q, (2, 2, 7, 9)))


@given(seed=st.integers(0, 2**31 - 1), s=st.integers(1, 48))
def test_mask_mod_equals_native_causal(seed, s):
    rng = np.random.default_rng(seed)
    q, k, v = random_qkv(rng, 2, s, s, 2, 2, 16)
    native = flash_attn(q, k, v, causal=True, backend="reference")
    flex = flash_attn(q, k, v, mask_mod=causal_mask(), backend="reference")
    assert_close(flex, native, dtype=q.dtype)


@given(
    seed=st.integers(0, 2**31 - 1),
    s=st.integers(1, 48),
    left=st.integers(0, 50),
    right=st.integers(0, 50),
)
def test_mask_mod_equals_native_window(seed, s, left, right):
    rng = np.random.default_rng(seed)
    q, k, v = random_qkv(rng, 1, s, s, 2, 2, 16)
    native = flash_attn(q, k, v, window_size=(left, right), backend="reference")
    flex = flash_attn(q, k, v, mask_mod=sliding_window_mask(left, right), backend="reference")
    assert_close(flex, native, dtype=q.dtype)


def test_mask_mod_combines_with_native_options():
    """mask_mod ANDs with causal/window rather than replacing them."""
    rng = np.random.default_rng(3)
    s = 32
    q, k, v = random_qkv(rng, 1, s, s, 2, 2, 16)
    got = flash_attn(q, k, v, causal=True, mask_mod=sliding_window_mask(4, None), backend="reference")
    want = flash_attn(q, k, v, window_size=(4, 0), backend="reference")
    assert_close(got, want, dtype=q.dtype)


def test_document_mask_attention():
    """Block-diagonal attention equals attending within each segment separately."""
    rng = np.random.default_rng(4)
    lens = [5, 9, 2]
    s = sum(lens)
    q, k, v = random_qkv(rng, 1, s, s, 2, 2, 16)
    ids = jnp.asarray(np.repeat(np.arange(len(lens)), lens))[None]
    got = flash_attn(q, k, v, mask_mod=document_mask(), aux_tensors=(ids,), backend="reference")
    lo = 0
    for n in lens:
        seg = slice(lo, lo + n)
        want = reference_attn(q[:, seg], k[:, seg], v[:, seg])
        assert_close(got[:, seg], want, dtype=q.dtype, msg=f"segment {seg}")
        lo += n


def test_aux_values_are_runtime_data():
    """Two different aux contents give different outputs with one MaskMod."""
    rng = np.random.default_rng(5)
    s = 12
    q, k, v = random_qkv(rng, 1, s, s, 1, 1, 8)
    mm = document_mask()
    ids1 = jnp.asarray(np.repeat([0, 1], [6, 6]))[None]
    ids2 = jnp.asarray(np.repeat([0, 1], [3, 9]))[None]
    o1 = flash_attn(q, k, v, mask_mod=mm, aux_tensors=(ids1,), backend="reference")
    o2 = flash_attn(q, k, v, mask_mod=mm, aux_tensors=(ids2,), backend="reference")
    assert np.abs(np.asarray(o1 - o2, dtype=np.float32)).max() > 1e-3
