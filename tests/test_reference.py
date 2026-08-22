"""reference_attn (and the reference backend behind flash_attn) against
jax.nn.dot_product_attention, which is an entirely independent implementation
(XLA's), over many hypothesis-generated shapes.

Everything jax.nn can express is compared here: plain/causal/GQA/scale/
two-sided windows and the logsumexp residual. Semantics jax.nn cannot express
(mask mods, empty-row zeros) are covered in test_masks.py.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from conftest import assert_close, random_qkv
from hypothesis import given, settings
from hypothesis import strategies as st

from fa4_jax import flash_attn, reference_attn

shapes = st.fixed_dictionaries(
    dict(
        b=st.integers(1, 3),
        s_q=st.integers(1, 65),
        s_kv=st.integers(1, 65),
        h_kv=st.integers(1, 3),
        gqa=st.sampled_from([1, 1, 2, 4]),
        d=st.sampled_from([8, 16, 24, 32]),
        seed=st.integers(0, 2**31 - 1),
    )
)


def _sdpa(q, k, v, **kw):
    # jax.nn.dot_product_attention supports GQA directly (broadcasts kv heads).
    return jax.nn.dot_product_attention(q, k, v, **kw)


@given(cfg=shapes, causal=st.booleans())
def test_matches_jax_nn(cfg, causal):
    rng = np.random.default_rng(cfg["seed"])
    q, k, v = random_qkv(
        rng, cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["h_kv"] * cfg["gqa"], cfg["h_kv"],
        cfg["d"],
    )
    if causal and cfg["s_q"] != cfg["s_kv"]:
        # jax.nn and flash-attention disagree on causal alignment for s_q !=
        # s_kv (top-left vs bottom-right); skip the comparison rather than
        # encode either convention here.
        return
    got = reference_attn(q, k, v, causal=causal)
    want = _sdpa(q, k, v, is_causal=causal)
    assert_close(got, want, dtype=q.dtype)


@given(cfg=shapes, scale=st.floats(0.05, 2.0))
def test_matches_jax_nn_scale(cfg, scale):
    rng = np.random.default_rng(cfg["seed"])
    q, k, v = random_qkv(
        rng, cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["h_kv"] * cfg["gqa"], cfg["h_kv"],
        cfg["d"],
    )
    got = reference_attn(q, k, v, scale=scale)
    want = _sdpa(q, k, v, scale=scale)
    assert_close(got, want, dtype=q.dtype)


@given(cfg=shapes, left=st.integers(0, 70), right=st.integers(0, 70))
def test_matches_jax_nn_window(cfg, left, right):
    # local windows are a self-attention concept; keep s_q == s_kv
    rng = np.random.default_rng(cfg["seed"])
    s = cfg["s_q"]
    q, k, v = random_qkv(
        rng, cfg["b"], s, s, cfg["h_kv"] * cfg["gqa"], cfg["h_kv"], cfg["d"]
    )
    got = reference_attn(q, k, v, window_size=(left, right))
    want = _sdpa(q, k, v, local_window_size=(left, right))
    assert_close(got, want, dtype=q.dtype)


@given(cfg=shapes)
def test_lse_matches_jax_nn(cfg):
    rng = np.random.default_rng(cfg["seed"])
    q, k, v = random_qkv(
        rng, cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["h_kv"] * cfg["gqa"], cfg["h_kv"],
        cfg["d"],
    )
    _, lse = reference_attn(q, k, v, return_lse=True)
    _, want = _sdpa(q, k, v, return_residual=True)  # (b, s, h), natural log
    assert_close(lse.transpose(0, 2, 1), want, dtype=q.dtype)


@given(cfg=shapes, causal=st.booleans())
def test_flash_attn_reference_backend_dispatch(cfg, causal):
    """flash_attn(backend="reference") is the primitive-wrapped reference."""
    rng = np.random.default_rng(cfg["seed"])
    s = cfg["s_q"]
    q, k, v = random_qkv(
        rng, cfg["b"], s, s, cfg["h_kv"] * cfg["gqa"], cfg["h_kv"], cfg["d"]
    )
    got = flash_attn(q, k, v, causal=causal, backend="reference")
    want = reference_attn(q, k, v, causal=causal)
    # not bit-identical: the primitive impl runs under jit, the reference
    # eagerly, and XLA fusion changes rounding at the last ulp.
    assert_close(got, want, dtype=q.dtype)


@settings(max_examples=10)
@given(cfg=shapes)
def test_jit_and_eager_agree(cfg):
    rng = np.random.default_rng(cfg["seed"])
    q, k, v = random_qkv(
        rng, cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["h_kv"] * cfg["gqa"], cfg["h_kv"], cfg["d"]
    )
    f = lambda q, k, v: flash_attn(q, k, v, causal=True, backend="reference")
    assert_close(jax.jit(f)(q, k, v), f(q, k, v), dtype=q.dtype)


def test_empty_row_convention():
    """Rows with no visible key give zeros and lse=-inf (flash convention)."""
    rng = np.random.default_rng(0)
    q, k, v = random_qkv(rng, 1, 6, 6, 2, 2, 16)
    # window (-2, 3): rows near the end see nothing (q - kv <= -2 needs kv >= q+2)
    out, lse = reference_attn(q, k, v, window_size=(-2, 3), return_lse=True)
    assert not np.any(np.isnan(np.asarray(out)))
    np.testing.assert_array_equal(np.asarray(out[0, 4:]), 0.0)
    assert np.all(np.isneginf(np.asarray(lse[0, :, 4:])))
    # visible rows are genuine attention outputs
    assert np.all(np.isfinite(np.asarray(lse[0, :, :4])))


def test_scale_default_is_rsqrt_head_dim():
    rng = np.random.default_rng(1)
    q, k, v = random_qkv(rng, 1, 8, 8, 1, 1, 32)
    assert_close(
        reference_attn(q, k, v),
        reference_attn(q, k, v, scale=1.0 / math.sqrt(32)),
        atol=0,
        rtol=0,
    )


def test_half_dtypes_roundtrip_dtype():
    rng = np.random.default_rng(2)
    for dtype in (jnp.bfloat16, jnp.float16):
        q, k, v = random_qkv(rng, 1, 16, 16, 2, 2, 16, dtype=dtype)
        out = reference_attn(q, k, v, causal=True)
        assert out.dtype == dtype
        want = _sdpa(
            q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32), is_causal=True
        )
        assert_close(out, want, dtype=dtype)


def test_head_dim_v_differs():
    """jax.nn can't express d_v != d; check against a plain einsum softmax."""
    rng = np.random.default_rng(6)
    q, k, v = random_qkv(rng, 2, 12, 15, 2, 2, 16, d_v=24)
    out = reference_attn(q, k, v)
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(16)
    p = jax.nn.softmax(s, axis=-1)
    want = jnp.einsum("bhqk,bkhd->bqhd", p, v)
    assert out.shape == (2, 12, 2, 24)
    assert_close(out, want, dtype=q.dtype)
