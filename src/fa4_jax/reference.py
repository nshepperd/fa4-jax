"""Pure-JAX reference semantics for fa4-jax.

This is the executable specification the CuTe kernels are tested against, and
doubles as the "reference" backend so every piece of the differentiation
wiring (custom primitives, JVP rule, transpose rule) runs — and is tested —
against ordinary JAX autodiff.

The math is ``jax.nn.dot_product_attention(implementation="xla")``, an
implementation independent of anything in this repo; the only local work is
mapping conventions onto it:

- Layout ``(batch, seqlen, heads, head_dim)``; GQA by giving k/v fewer heads.
- ``window_size=(left, right)`` counted from the query position, inclusive
  band ``-right <= q - kv <= left``, ``None`` meaning unbounded; combines
  with ``causal`` and with a ``mask_mod`` (which becomes a dense boolean
  mask).
- Rows with no visible key return **zeros** (and ``lse = -inf``), the
  flash-attention convention; jax.nn would return a uniform average with a
  huge negative lse instead, so those rows are patched afterwards.
- ``lse`` is the natural-log row log-sum-exp of the *scaled* scores, laid
  out ``(batch, heads, q_seq)``.

Everything computes in float32 and casts back to the input dtype at the end.
"""

from __future__ import annotations

import math

import jax
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
        mm = mask_mod(b, h, q[None, None], kv[None, None])
        mm = jnp.broadcast_to(mm, (batch, heads, s_q, s_kv))
        mask = mm if mask is None else mask & mm
    return mask


def fwd_lse(q, k, v, scale: float, causal: bool, window, mask_mod: MaskMod | None):
    """(out, lse) in float32 from float32 inputs; see the module docstring.

    ``causal``/``window`` go to jax.nn natively (its band logic is an
    independent check of ours); the mask_mod rides in as a dense mask. The
    combined dense mask is still built, to find rows with no visible key.
    """
    b, s_q, h, d = q.shape
    s_kv, d_v = k.shape[1], v.shape[3]
    left, right = window
    local = left is not None or right is not None
    mod_mask = dense_mask(b, h, s_q, s_kv, False, (None, None), mask_mod)

    # jax.nn wants one head dim for q/k/v: zero-pad to the larger (padding
    # q/k leaves the scores unchanged; padded v columns are sliced off).
    dd = max(d, d_v)
    pad = lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, dd - x.shape[3])))
    out = jax.nn.dot_product_attention(
        pad(q),
        pad(k),
        pad(v),
        mask=mod_mask,
        scale=scale,
        is_causal=causal,
        local_window_size=(
            s_kv if left is None else left,
            s_kv if right is None else right,
        )
        if local
        else None,
        implementation="xla",
    )[..., :d_v]

    # jax.nn's return_residual lse is stop_gradient'ed, so take the lse from
    # the scores directly (jax.nn.logsumexp; differentiable, -inf on empty
    # rows with a finite gradient).
    full = dense_mask(b, h, s_q, s_kv, causal, window, mask_mod)
    s = jnp.einsum("bqhd,bkhd->bhqk", q, _expand_kv(k, h)) * scale
    if full is not None:
        s = jnp.where(full, s, -jnp.inf)
        empty = ~full.any(-1)  # (b, h, s_q)
        out = jnp.where(jnp.moveaxis(empty, 1, 2)[..., None], 0.0, out)
    lse = jax.nn.logsumexp(s, axis=-1)
    return out, lse


def _expand_kv(x: Array, q_heads: int) -> Array:
    """Repeat kv heads up to q heads (GQA)."""
    kv_heads = x.shape[2]
    if kv_heads == q_heads:
        return x
    return jnp.repeat(x, q_heads // kv_heads, axis=2)


def reference_attn(
    q: Float[Array, "batch q_seq q_heads head_dim"],
    k: Float[Array, "batch kv_seq kv_heads head_dim"],
    v: Float[Array, "batch kv_seq kv_heads head_dim_v"],
    *,
    causal: bool = False,
    window_size: tuple[int | None, int | None] | None = None,
    scale: float | None = None,
    mask_mod: MaskMod | None = None,
    return_lse: bool = False,
):
    """Reference attention with the exact semantics of ``fa4_jax.flash_attn``.

    Differentiable by ordinary JAX autodiff (it is plain jnp), so it provides
    independent ground truth for values, JVPs, and VJPs alike.
    """
    scale = float(scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    window = window_size if window_size is not None else (None, None)
    f32 = jnp.float32
    out, lse = fwd_lse(
        q.astype(f32), k.astype(f32), v.astype(f32), scale, causal, window, mask_mod
    )
    out = out.astype(q.dtype)
    return (out, lse) if return_lse else out
