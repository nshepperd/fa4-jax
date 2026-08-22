"""The plain-jnp backend: reference semantics behind the Backend protocol.

Used on CPU (and anywhere ``backend="reference"`` is asked for), and by the
unit tests to exercise the primitive/differentiation wiring without a GPU.
GQA is handled here by expanding kv heads before the math and summing
gradients back down after; the cute backend instead bakes the head ratio into
the kernels.
"""

from __future__ import annotations

import jax.numpy as jnp

from fa4_jax import reference as ref


class ReferenceBackend:
    def _mask(self, cfg, q, k, aux):
        return ref.dense_mask(
            q.shape[0], q.shape[2], q.shape[1], k.shape[1],
            cfg.causal, cfg.window, cfg.mask_mod, aux,
        )

    def _expand(self, cfg, q, k, v):
        h_q = q.shape[2]
        return (
            q.astype(jnp.float32),
            ref._expand_kv(k, h_q).astype(jnp.float32),
            ref._expand_kv(v, h_q).astype(jnp.float32),
        )

    def fwd(self, cfg, q, k, v, aux):
        return self.fwd_lse(cfg, q, k, v, aux)[0]

    def fwd_lse(self, cfg, q, k, v, aux):
        qf, kf, vf = self._expand(cfg, q, k, v)
        out, lse = ref.fwd_lse(qf, kf, vf, cfg.scale, self._mask(cfg, q, k, aux))
        return out.astype(q.dtype), lse

    def tangent(self, cfg, q, k, v, out, lse, dq, dk, dv, aux):
        qf, kf, vf = self._expand(cfg, q, k, v)
        h_q = q.shape[2]
        dqf = dq.astype(jnp.float32)
        dkf = ref._expand_kv(dk, h_q).astype(jnp.float32)
        dvf = ref._expand_kv(dv, h_q).astype(jnp.float32)
        t_out, t_lse = ref.tangent(
            qf, kf, vf, out.astype(jnp.float32), lse, dqf, dkf, dvf,
            cfg.scale, self._mask(cfg, q, k, aux),
        )
        return t_out.astype(out.dtype), t_lse

    def vjp(self, cfg, q, k, v, out, lse, d_out, d_lse, aux):
        qf, kf, vf = self._expand(cfg, q, k, v)
        dq, dk, dv = ref.vjp(
            qf, kf, vf, out.astype(jnp.float32), lse, d_out.astype(jnp.float32),
            d_lse, cfg.scale, self._mask(cfg, q, k, aux),
        )
        kv_heads = k.shape[2]
        dk = ref._fold_kv(dk, kv_heads)
        dv = ref._fold_kv(dv, kv_heads)
        return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
