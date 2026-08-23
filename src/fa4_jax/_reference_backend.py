"""The plain-jnp backend: reference semantics behind the Backend protocol.

Used anywhere ``backend="reference"`` is asked for, and by the tests as the
oracle the kernels are compared against. The tangent and VJP are ordinary
``jax.jvp``/``jax.vjp`` of the forward (recomputing it — this is the
reference, cost is not the point), so the custom primitives' differentiation
rules are checked against JAX's own autodiff rather than against a second
hand derivation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from fa4_jax import reference as ref


def _fwd(cfg, aux):
    """The forward as a function of (q, k, v) alone; the mask's array leaves
    are closed over (they are never differentiated — flash_attn's JVP rule
    rejects their tangents)."""
    f32 = jnp.float32
    mask_mod = cfg.mask_mod(aux)

    def f(q, k, v):
        return ref.fwd_lse(
            q.astype(f32),
            k.astype(f32),
            v.astype(f32),
            cfg.scale,
            cfg.causal,
            cfg.window,
            mask_mod,
        )

    return f


class ReferenceBackend:
    def fwd(self, cfg, q, k, v, aux):
        return self.fwd_lse(cfg, q, k, v, aux)[0]

    def fwd_lse(self, cfg, q, k, v, aux):
        out, lse = _fwd(cfg, aux)(q, k, v)
        return out.astype(q.dtype), lse

    def tangent(self, cfg, q, k, v, out, lse, dq, dk, dv, aux):
        _, (t_out, t_lse) = jax.jvp(_fwd(cfg, aux), (q, k, v), (dq, dk, dv))
        return t_out.astype(out.dtype), t_lse

    def vjp(self, cfg, q, k, v, out, lse, d_out, d_lse, aux):
        (_, lse_p), pull = jax.vjp(_fwd(cfg, aux), q, k, v)
        d_lse = jnp.zeros_like(lse_p) if d_lse is None else d_lse
        dq, dk, dv = pull((d_out.astype(jnp.float32), d_lse))
        return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
