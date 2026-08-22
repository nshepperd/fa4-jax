"""Differentiation wiring: flash_attn is built on custom primitives (a JVP
rule plus a linear tangent primitive with a transpose rule), and on the
reference backend those primitives wrap the same math as reference_attn —
so autodiff of flash_attn must agree with *plain JAX autodiff* of
reference_attn in every mode. That checks the custom tangent/vjp formulas
and the transpose bookkeeping, not just the forward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_close, random_qkv
from hypothesis import given
from hypothesis import strategies as st

from fa4_jax import flash_attn, reference_attn, sliding_window_mask

shapes = st.fixed_dictionaries(
    dict(
        b=st.integers(1, 2),
        s_q=st.integers(1, 33),
        s_kv=st.integers(1, 33),
        h_kv=st.integers(1, 2),
        gqa=st.sampled_from([1, 2]),
        d=st.sampled_from([8, 16]),
        seed=st.integers(0, 2**31 - 1),
    )
)


def _pair(cfg, causal=False, window=None, **kw):
    """(f, g): identical semantics; f via primitives, g via plain jnp autodiff."""
    f = lambda q, k, v: flash_attn(
        q, k, v, causal=causal, window_size=window, backend="reference", **kw
    )
    g = lambda q, k, v: reference_attn(q, k, v, causal=causal, window_size=window, **kw)
    return f, g


def _rand_like(rng, *arrays):
    return tuple(jnp.asarray(rng.standard_normal(a.shape), a.dtype) for a in arrays)


@given(cfg=shapes, causal=st.booleans())
def test_jvp_matches_plain_autodiff(cfg, causal):
    rng = np.random.default_rng(cfg["seed"])
    q, k, v = random_qkv(
        rng, cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["h_kv"] * cfg["gqa"], cfg["h_kv"], cfg["d"]
    )
    dq, dk, dv = _rand_like(rng, q, k, v)
    f, g = _pair(cfg, causal=causal)
    out_f, tan_f = jax.jvp(f, (q, k, v), (dq, dk, dv))
    out_g, tan_g = jax.jvp(g, (q, k, v), (dq, dk, dv))
    assert_close(out_f, out_g, dtype=q.dtype)
    assert_close(tan_f, tan_g, dtype=q.dtype)


@given(cfg=shapes, causal=st.booleans())
def test_grad_matches_plain_autodiff(cfg, causal):
    rng = np.random.default_rng(cfg["seed"])
    q, k, v = random_qkv(
        rng, cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["h_kv"] * cfg["gqa"], cfg["h_kv"], cfg["d"]
    )
    cot = jnp.asarray(rng.standard_normal((cfg["b"], cfg["s_q"], cfg["h_kv"] * cfg["gqa"], cfg["d"])), q.dtype)
    f, g = _pair(cfg, causal=causal)
    grads_f = jax.grad(lambda *a: (f(*a) * cot).sum(), argnums=(0, 1, 2))(q, k, v)
    grads_g = jax.grad(lambda *a: (g(*a) * cot).sum(), argnums=(0, 1, 2))(q, k, v)
    for got, want, name in zip(grads_f, grads_g, "qkv"):
        assert_close(got, want, dtype=q.dtype, msg=f"d{name}")


@given(cfg=shapes)
def test_window_grad_and_jvp(cfg):
    rng = np.random.default_rng(cfg["seed"])
    s = cfg["s_q"]
    q, k, v = random_qkv(rng, cfg["b"], s, s, cfg["h_kv"], cfg["h_kv"], cfg["d"])
    dq, dk, dv = _rand_like(rng, q, k, v)
    f, g = _pair(cfg, window=(3, 5))
    _, tan_f = jax.jvp(f, (q, k, v), (dq, dk, dv))
    _, tan_g = jax.jvp(g, (q, k, v), (dq, dk, dv))
    assert_close(tan_f, tan_g, dtype=q.dtype)
    gf = jax.grad(lambda *a: f(*a).sum(), argnums=(0, 1, 2))(q, k, v)
    gg = jax.grad(lambda *a: g(*a).sum(), argnums=(0, 1, 2))(q, k, v)
    for got, want in zip(gf, gg):
        assert_close(got, want, dtype=q.dtype)


def test_mask_mod_grad_and_jvp():
    rng = np.random.default_rng(7)
    q, k, v = random_qkv(rng, 2, 24, 24, 2, 2, 16)
    dq, dk, dv = _rand_like(rng, q, k, v)
    mm = sliding_window_mask(4, 2)
    f = lambda *a: flash_attn(*a, mask_mod=mm, backend="reference")
    g = lambda *a: reference_attn(*a, mask_mod=mm)
    _, tan_f = jax.jvp(f, (q, k, v), (dq, dk, dv))
    _, tan_g = jax.jvp(g, (q, k, v), (dq, dk, dv))
    assert_close(tan_f, tan_g, dtype=q.dtype)
    gf = jax.grad(lambda *a: f(*a).sum(), argnums=(0, 1, 2))(q, k, v)
    gg = jax.grad(lambda *a: g(*a).sum(), argnums=(0, 1, 2))(q, k, v)
    for got, want in zip(gf, gg):
        assert_close(got, want, dtype=q.dtype)


@given(cfg=shapes)
def test_lse_is_differentiable(cfg):
    """return_lse=True: grad and jvp flow through the lse output too (the
    tangent primitive's t_lse = mu and the d_lse path of the transpose)."""
    rng = np.random.default_rng(cfg["seed"])
    q, k, v = random_qkv(
        rng, cfg["b"], cfg["s_q"], cfg["s_kv"], cfg["h_kv"], cfg["h_kv"], cfg["d"]
    )
    dq, dk, dv = _rand_like(rng, q, k, v)

    def f(q, k, v):
        out, lse = flash_attn(q, k, v, return_lse=True, backend="reference")
        return out.sum() + (lse * 0.7).sum()

    def g(q, k, v):
        out, lse = reference_attn(q, k, v, return_lse=True)
        return out.sum() + (lse * 0.7).sum()

    _, tan_f = jax.jvp(f, (q, k, v), (dq, dk, dv))
    _, tan_g = jax.jvp(g, (q, k, v), (dq, dk, dv))
    assert_close(tan_f, tan_g, atol=1e-3, rtol=1e-3)
    gf = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
    gg = jax.grad(g, argnums=(0, 1, 2))(q, k, v)
    for got, want in zip(gf, gg):
        assert_close(got, want, dtype=q.dtype)


def test_jacfwd_equals_jacrev():
    """Forward and reverse jacobians of the primitive agree (tiny problem)."""
    rng = np.random.default_rng(11)
    q, k, v = random_qkv(rng, 1, 5, 5, 1, 1, 8)
    f = lambda q: flash_attn(q, k, v, causal=True, backend="reference")
    jf = jax.jacfwd(f)(q)
    jr = jax.jacrev(f)(q)
    assert_close(jf, jr, atol=1e-5, rtol=1e-5)


def test_linearize_transpose_roundtrip():
    """<J u, w> == <u, J^T w>: the transpose rule is the true adjoint of the
    tangent kernel (dot-product test, independent of any reference)."""
    rng = np.random.default_rng(13)
    q, k, v = random_qkv(rng, 2, 17, 19, 2, 2, 16)
    f = lambda q, k, v: flash_attn(q, k, v, causal=False, backend="reference")
    out, f_jvp = jax.linearize(f, q, k, v)
    u = _rand_like(np.random.default_rng(14), q, k, v)
    w = jnp.asarray(np.random.default_rng(15).standard_normal(out.shape), out.dtype)
    Ju = f_jvp(*u)
    JTw = jax.linear_transpose(f_jvp, q, k, v)(w)
    lhs = jnp.vdot(Ju, w)
    rhs = sum(jnp.vdot(a, b) for a, b in zip(u, JTw))
    assert_close(lhs, rhs, atol=1e-3, rtol=1e-3)


def test_zero_tangents():
    rng = np.random.default_rng(17)
    q, k, v = random_qkv(rng, 1, 9, 9, 1, 1, 8)
    f = lambda k: flash_attn(q, k, v, backend="reference")
    dk = jnp.asarray(rng.standard_normal(k.shape), k.dtype)
    _, tan_f = jax.jvp(f, (k,), (dk,))
    g = lambda k: reference_attn(q, k, v)
    _, tan_g = jax.jvp(g, (k,), (dk,))
    assert_close(tan_f, tan_g, dtype=q.dtype)


def test_aux_tensors_not_differentiable():
    rng = np.random.default_rng(19)
    q, k, v = random_qkv(rng, 1, 6, 6, 1, 1, 8)
    ids = jnp.zeros((1, 6), jnp.float32)  # float so jax will try to differentiate
    from fa4_jax import document_mask

    f = lambda ids: flash_attn(
        q, k, v, mask_mod=document_mask(), aux_tensors=(ids,), backend="reference"
    ).sum()
    with pytest.raises(NotImplementedError, match="aux"):
        jax.grad(f)(ids)


def test_grad_under_jit():
    rng = np.random.default_rng(23)
    q, k, v = random_qkv(rng, 1, 16, 16, 2, 2, 16)
    f = lambda q, k, v: flash_attn(q, k, v, causal=True, backend="reference").sum()
    g = lambda q, k, v: reference_attn(q, k, v, causal=True).sum()
    got = jax.jit(jax.grad(f, argnums=(0, 1, 2)))(q, k, v)
    want = jax.grad(g, argnums=(0, 1, 2))(q, k, v)
    for a, b in zip(got, want):
        assert_close(a, b, dtype=q.dtype)


def test_gqa_grad_folds_heads():
    """GQA gradients: dk/dv come back at kv-head width, equal to summing the
    expanded-head gradient over each group."""
    rng = np.random.default_rng(29)
    q, k, v = random_qkv(rng, 1, 12, 12, 4, 2, 8)
    f = lambda k, v: flash_attn(q, k, v, backend="reference").sum()
    dk, dv = jax.grad(f, argnums=(0, 1))(k, v)
    assert dk.shape == k.shape and dv.shape == v.shape
    kx = jnp.repeat(k, 2, axis=2)
    vx = jnp.repeat(v, 2, axis=2)
    g = lambda kx, vx: reference_attn(q, kx, vx).sum()
    dkx, dvx = jax.grad(g, argnums=(0, 1))(kx, vx)
    assert_close(dk, dkx.reshape(1, 12, 2, 2, 8).sum(3), dtype=q.dtype)
    assert_close(dv, dvx.reshape(1, 12, 2, 2, 8).sum(3), dtype=q.dtype)
