"""Cheap API-level checks: input validation, window canonicalization, and the
MaskMod pytree contract (what keys a kernel vs. what is runtime data). These
run on the cute backend where a kernel is involved; the rest never reaches a
kernel at all."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_close, random_qkv

from fa4_jax import (
    FnMask,
    MaskMod,
    and_masks,
    causal_mask,
    document_mask,
    flash_attn,
    or_masks,
    sliding_window_mask,
    static,
)
from fa4_jax.attention import _resolve_causal_local_window
from fa4_jax.reference import dense_mask


def _qkv(dtype=jnp.bfloat16):
    return random_qkv(np.random.default_rng(0), 1, 64, 64, 2, 2, 64, dtype=dtype)


# ---------------------------------------------------------------- validation


def test_rejects_bad_rank():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="batch, seqlen, heads, head_dim"):
        flash_attn(q[0], k, v)


def test_rejects_dtype_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="dtypes must match"):
        flash_attn(q, k.astype(jnp.float16), v)


def test_rejects_bad_gqa_ratio():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="not a multiple"):
        flash_attn(q[:, :, :1].repeat(3, 2), k, v)


def test_rejects_kv_shape_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="k/v"):
        flash_attn(q, k, v[:, :4])


def test_rejects_head_dim_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="head_dim mismatch"):
        flash_attn(q[..., :32], k, v)


def test_rejects_raw_callable_mask():
    q, k, v = _qkv()
    with pytest.raises(TypeError, match="MaskMod"):
        flash_attn(q, k, v, mask_mod=lambda b, h, qi, ki: ki <= qi)


def test_rejects_f32_on_cute():
    q, k, v = _qkv(jnp.float32)
    with pytest.raises(ValueError, match="fp16/bf16"):
        flash_attn(q, k, v, backend="cute")


def test_unknown_backend():
    q, k, v = _qkv()
    with pytest.raises(KeyError):
        flash_attn(q, k, v, backend="nope")


def test_window_canonicalization():
    # bare right-window 0 is the causal kernel
    assert _resolve_causal_local_window(False, None, 0) == (True, False, None, None)
    # causal flag forces right=0 and stays causal
    assert _resolve_causal_local_window(True, None, None) == (True, False, None, None)
    # causal + left window becomes the local kernel with right=0
    assert _resolve_causal_local_window(True, 5, None) == (False, True, 5, 0)
    # a window that excludes everything is dropped entirely (upstream rule)
    assert _resolve_causal_local_window(False, 2, -5) == (False, False, None, None)
    # plain two-sided window
    assert _resolve_causal_local_window(False, 3, 4) == (False, True, 3, 4)


def test_causal_flag_equals_right_zero_window():
    q, k, v = _qkv()
    a = flash_attn(q, k, v, causal=True)
    b = flash_attn(q, k, v, window_size=(None, 0))
    assert_close(a, b, atol=0, rtol=0)


def test_causal_and_window_combine():
    q, k, v = _qkv()
    a = flash_attn(q, k, v, causal=True, window_size=(3, None))
    b = flash_attn(q, k, v, window_size=(3, 0))
    assert_close(a, b, atol=0, rtol=0)


# ---------------------------------------------------------------- mask pytrees


def _dense(mask, b=2, h=3, s_q=7, s_kv=9):
    return np.asarray(dense_mask(b, h, s_q, s_kv, False, (None, None), mask))


def test_dense_mask_semantics():
    q, kv = np.mgrid[0:7, 0:9]
    np.testing.assert_array_equal(_dense(causal_mask()), np.broadcast_to(kv <= q, (2, 3, 7, 9)))
    want = (q - kv <= 2) & (kv - q <= 1)
    np.testing.assert_array_equal(_dense(sliding_window_mask(2, 1)), np.broadcast_to(want, (2, 3, 7, 9)))
    a, b = causal_mask(), sliding_window_mask(2, None)
    np.testing.assert_array_equal(_dense(and_masks(a, b)), _dense(a) & _dense(b))
    np.testing.assert_array_equal(_dense(or_masks(a, b)), _dense(a) | _dense(b))
    head = FnMask(lambda b, h, q, kv: (kv <= q) | (h == 0), label="h0-full")
    d = _dense(head)
    assert d[:, 0].all()
    np.testing.assert_array_equal(d[:, 1:], np.broadcast_to(kv <= q, (2, 2, 7, 9)))


def test_document_mask_dense():
    ids = jnp.asarray([[0, 0, 1, 1, 1, 2, 2, 2, 2]])
    m = _dense(document_mask(ids), b=1, s_q=9, s_kv=9)
    want = np.asarray(ids[0])[:, None] == np.asarray(ids[0])[None, :]
    np.testing.assert_array_equal(m, np.broadcast_to(want, (1, 3, 9, 9)))


def test_static_fields_key_the_treedef():
    """Static fields distinguish kernels; array fields don't; the treedef
    round-trips to a working callable (the path the cute backend takes)."""
    a, b, c = sliding_window_mask(2, 1), sliding_window_mask(2, 1), sliding_window_mask(3, 1)
    assert jax.tree.structure(a) == jax.tree.structure(b)
    assert jax.tree.structure(a) != jax.tree.structure(c)
    assert jax.tree.leaves(a) == []

    ids1 = jnp.asarray([[0, 0, 1]])
    ids2 = jnp.asarray([[0, 1, 1]])
    assert jax.tree.structure(document_mask(ids1)) == jax.tree.structure(document_mask(ids2))

    full = and_masks(causal_mask(), document_mask(ids1))
    leaves, treedef = jax.tree.flatten(full)
    assert len(leaves) == 1
    rebuilt = treedef.unflatten(leaves)
    np.testing.assert_array_equal(_dense(rebuilt, b=1, s_q=3, s_kv=3), _dense(full, b=1, s_q=3, s_kv=3))


def test_custom_mask_class():
    """User-defined MaskMod subclasses register as pytrees automatically and
    run on the kernels."""

    @dataclass(frozen=True)
    class PrefixMask(MaskMod):
        prefix_len: jax.Array  # (batch,) runtime
        stride: int = static(default=1)

        def __call__(self, b, h, q, kv):
            return (kv < self.prefix_len[b]) | ((kv <= q) & (kv % self.stride == 0))

    m = PrefixMask(jnp.asarray([2, 4]), stride=2)
    assert jax.tree.leaves(m) == [m.prefix_len]
    d = _dense(m, b=2, h=1, s_q=6, s_kv=6)
    q, kv = np.mgrid[0:6, 0:6]
    for bi, p in enumerate([2, 4]):
        np.testing.assert_array_equal(d[bi, 0], (kv < p) | ((kv <= q) & (kv % 2 == 0)))

    q, k, v = random_qkv(np.random.default_rng(1), 2, 96, 96, 1, 1, 64, dtype=jnp.bfloat16)
    m = PrefixMask(jnp.asarray([16, 40], jnp.int32), stride=3)
    got = flash_attn(q, k, v, mask_mod=m, backend="cute")
    want = flash_attn(q, k, v, mask_mod=m, backend="reference")
    assert_close(got, want, dtype=q.dtype)


def test_non_array_leaf_rejected():
    @dataclass(frozen=True)
    class Bad(MaskMod):
        left: int  # forgot static()

        def __call__(self, b, h, q, kv):
            return q - kv <= self.left

    q, k, v = _qkv()
    with pytest.raises(TypeError, match="static"):
        flash_attn(q, k, v, mask_mod=Bad(3))


def test_mask_arrays_not_differentiable():
    q, k, v = _qkv()
    ids = jnp.zeros((1, 64), jnp.float32)  # float so jax will try to differentiate
    f = lambda ids: flash_attn(q, k, v, mask_mod=document_mask(ids)).sum()
    with pytest.raises(NotImplementedError, match="aux"):
        jax.grad(f)(ids)
