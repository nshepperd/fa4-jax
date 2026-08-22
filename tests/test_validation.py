"""Input validation and canonicalization behavior of the public API."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_close, random_qkv

from fa4_jax import flash_attn, reference_attn
from fa4_jax.attention import _resolve_causal_local_window


def _qkv():
    return random_qkv(np.random.default_rng(0), 1, 8, 8, 2, 2, 16)


def test_rejects_bad_rank():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="batch, seqlen, heads, head_dim"):
        flash_attn(q[0], k, v, backend="reference")


def test_rejects_dtype_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="dtypes must match"):
        flash_attn(q, k.astype(jnp.bfloat16), v, backend="reference")


def test_rejects_bad_gqa_ratio():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="not a multiple"):
        flash_attn(q[:, :, :1].repeat(3, 2), k, v, backend="reference")


def test_rejects_kv_shape_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="k/v"):
        flash_attn(q, k, v[:, :4], backend="reference")


def test_rejects_head_dim_mismatch():
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="head_dim mismatch"):
        flash_attn(q[..., :8], k, v, backend="reference")


def test_rejects_raw_callable_mask():
    q, k, v = _qkv()
    with pytest.raises(TypeError, match="MaskMod"):
        flash_attn(q, k, v, mask_mod=lambda b, h, qi, ki, aux: ki <= qi, backend="reference")


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
    a = flash_attn(q, k, v, causal=True, backend="reference")
    b = flash_attn(q, k, v, window_size=(None, 0), backend="reference")
    assert_close(a, b, atol=0, rtol=0)


def test_causal_and_window_combine():
    q, k, v = _qkv()
    a = flash_attn(q, k, v, causal=True, window_size=(3, None), backend="reference")
    b = flash_attn(q, k, v, window_size=(3, 0), backend="reference")
    assert_close(a, b, atol=0, rtol=0)


def test_unknown_backend():
    q, k, v = _qkv()
    with pytest.raises(KeyError):
        flash_attn(q, k, v, backend="nope")


def test_reference_attn_and_flash_attn_default_scale_agree():
    q, k, v = _qkv()
    assert_close(
        flash_attn(q, k, v, backend="reference"),
        reference_attn(q, k, v),
        atol=0,
        rtol=0,
    )
