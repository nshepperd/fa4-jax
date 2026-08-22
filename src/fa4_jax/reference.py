"""Pure-JAX reference semantics for fa4-jax.

This is the executable specification the CuTe kernels are tested against, and
doubles as the "reference" backend so every piece of the differentiation
wiring (custom primitives, JVP rule, transpose rule) runs — and is unit
tested — on machines with no GPU.

Conventions (matching the FA4 kernels):

- Layout ``(batch, seqlen, heads, head_dim)``; GQA by giving k/v fewer heads.
- ``window_size=(left, right)`` counted from the query position, inclusive
  band ``-right <= q - kv <= left``, ``None`` meaning unbounded; combines
  with ``causal`` and with a ``mask_mod``.
- Rows with no visible key return **zeros** (and ``lse = -inf``), the
  flash-attention convention, where a naive softmax would return NaN.
- ``lse`` is the natural-log row log-sum-exp of the *scaled* scores.

Everything computes in float32 and casts back to the input dtype at the end.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from fa4_jax.masks import MaskMod

__all__ = ["reference_attn"]


def dense_mask(
    batch: int,
    heads: int,
    s_q: int,
    s_kv: int,
    causal: bool,
    window: tuple[int | None, int | None],
    mask_mod: MaskMod | None,
    aux: tuple[Array, ...],
) -> Bool[Array, "batch heads s_q s_kv"] | None:
    """The combined visibility mask as a dense boolean, or None if all-visible."""
    q = jnp.arange(s_q)[:, None]
    kv = jnp.arange(s_kv)[None, :]
    parts = []
    if causal:
        parts.append(kv <= q)
    left, right = window
    if left is not None:
        parts.append(q - kv <= left)
    if right is not None:
        parts.append(kv - q <= right)
    mask = None
    for p in parts:
        p = jnp.broadcast_to(p, (s_q, s_kv))[None, None]
        mask = p if mask is None else mask & p
    if mask is not None:
        mask = jnp.broadcast_to(mask, (batch, heads, s_q, s_kv))
    if mask_mod is not None:
        b = jnp.arange(batch)[:, None, None, None]
        h = jnp.arange(heads)[None, :, None, None]
        mm = mask_mod.fn(b, h, q[None, None], kv[None, None], aux)
        mm = jnp.broadcast_to(mm, (batch, heads, s_q, s_kv))
        mask = mm if mask is None else mask & mm
    return mask


def _expand_kv(x: Array, q_heads: int) -> Array:
    """Repeat kv heads up to q heads (GQA)."""
    kv_heads = x.shape[2]
    if kv_heads == q_heads:
        return x
    return jnp.repeat(x, q_heads // kv_heads, axis=2)


def _fold_kv(x: Array, kv_heads: int) -> Array:
    """Sum a per-q-head gradient back down to the kv heads (GQA transpose)."""
    b, s, hq, d = x.shape
    if hq == kv_heads:
        return x
    return x.reshape(b, s, kv_heads, hq // kv_heads, d).sum(axis=3)


def _scores(q: Array, k: Array, scale: float, mask) -> Array:
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) * scale
    if mask is not None:
        s = jnp.where(mask, s, -jnp.inf)
    return s


def _probs(s: Array, lse: Array) -> Array:
    """P = exp(S - lse) with fully-masked rows (lse = -inf) giving zeros."""
    lse_safe = jnp.where(jnp.isfinite(lse), lse, 0.0)
    p = jnp.exp(s - lse_safe[..., None])
    return jnp.where(jnp.isfinite(s), p, 0.0)


def fwd_lse(q, k, v, scale, mask):
    """Forward returning (out, lse); float32 in, float32 out."""
    s = _scores(q, k, scale, mask)
    m = jnp.max(s, axis=-1)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.where(jnp.isfinite(s), jnp.exp(s - m_safe[..., None]), 0.0)
    l = jnp.sum(p, axis=-1)
    lse = jnp.where(l > 0, jnp.log(jnp.where(l > 0, l, 1.0)) + m_safe, -jnp.inf)
    out = jnp.einsum("bhqk,bkhd->bqhd", p, v) / jnp.maximum(l, 1.0e-37).transpose(0, 2, 1)[..., None]
    return out, lse


def tangent(q, k, v, out, lse, dq, dk, dv, scale, mask):
    """JVP tangent: linear in (dq, dk, dv), with (q, k, v, out, lse) residual.

    With P = exp(S - lse) and tS = scale * (dq k^T + q dk^T):

        mu     = rowsum(P * tS)              (= tangent of lse)
        t_out  = (P * tS) v + P dv - mu * out
    """
    s = _scores(q, k, scale, mask)
    p = _probs(s, lse)
    ts = jnp.einsum("bqhd,bkhd->bhqk", dq, k, preferred_element_type=jnp.float32)
    ts = ts + jnp.einsum("bqhd,bkhd->bhqk", q, dk, preferred_element_type=jnp.float32)
    ts = ts * scale
    pts = p * ts
    mu = jnp.sum(pts, axis=-1)
    t_out = jnp.einsum("bhqk,bkhd->bqhd", pts, v)
    t_out = t_out + jnp.einsum("bhqk,bkhd->bqhd", p, dv)
    t_out = t_out - mu.transpose(0, 2, 1)[..., None] * out
    return t_out, mu


def vjp(q, k, v, out, lse, d_out, d_lse, scale, mask):
    """Reverse mode: cotangents for (q, k, v) given cotangents of (out, lse).

        dP = d_out v^T
        D  = rowsum(out * d_out)
        dS = P * (dP - D + d_lse)
        dq = scale * dS k ;  dk = scale * dS^T q ;  dv = P^T d_out
    """
    s = _scores(q, k, scale, mask)
    p = _probs(s, lse)
    dp = jnp.einsum("bqhd,bkhd->bhqk", d_out, v, preferred_element_type=jnp.float32)
    d_row = jnp.sum(out * d_out, axis=-1).transpose(0, 2, 1)
    if d_lse is not None:
        d_row = d_row - d_lse
    ds = p * (dp - d_row[..., None])
    dq = scale * jnp.einsum("bhqk,bkhd->bqhd", ds, k)
    dk = scale * jnp.einsum("bhqk,bqhd->bkhd", ds, q)
    dv = jnp.einsum("bhqk,bqhd->bkhd", p, d_out)
    return dq, dk, dv


def reference_attn(
    q: Float[Array, "batch q_seq q_heads head_dim"],
    k: Float[Array, "batch kv_seq kv_heads head_dim"],
    v: Float[Array, "batch kv_seq kv_heads head_dim_v"],
    *,
    causal: bool = False,
    window_size: tuple[int | None, int | None] | None = None,
    scale: float | None = None,
    mask_mod: MaskMod | None = None,
    aux_tensors: tuple[Array, ...] = (),
    return_lse: bool = False,
):
    """Reference attention with the exact semantics of ``fa4_jax.flash_attn``.

    Differentiable by ordinary JAX autodiff (it is plain jnp), so it provides
    independent ground truth for values, JVPs, and VJPs alike.
    """
    b, s_q, h_q, _ = q.shape
    s_kv = k.shape[1]
    scale = float(scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    window = window_size if window_size is not None else (None, None)
    mask = dense_mask(b, h_q, s_q, s_kv, causal, window, mask_mod, aux_tensors)
    qf = q.astype(jnp.float32)
    kf = _expand_kv(k, h_q).astype(jnp.float32)
    vf = _expand_kv(v, h_q).astype(jnp.float32)
    out, lse = fwd_lse(qf, kf, vf, scale, mask)
    out = out.astype(q.dtype)
    return (out, lse) if return_lse else out
