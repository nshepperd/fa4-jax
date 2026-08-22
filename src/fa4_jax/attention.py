"""Public ``flash_attn`` and its differentiation wiring.

The function is built on two JAX primitives rather than ``jax.custom_vjp``,
because it supports *both* differentiation modes and custom_vjp forbids
forward mode:

- ``fa4_fwd``: ``(q, k, v, *aux) -> (out[, lse])``. Its JVP rule computes the
  primal via the kernel forward (with LSE) and the tangent via ``fa4_tangent``.
- ``fa4_tangent``: ``(q, k, v, out, lse, dq, dk, dv, *aux) -> (t_out, t_lse)``.
  Linear in ``(dq, dk, dv)``; its *transpose* rule is the FA4 backward pass.

Reverse mode therefore costs exactly what a custom_vjp would (linearization
runs the forward-with-LSE; transposition runs the backward kernels; the
tangent kernel itself never executes), while ``jax.jvp``/``jax.linearize``
run the forward plus the fused tangent kernel. ``t_lse`` is the true LSE
tangent (the row sum ``mu = rowsum(P * tS)``, free inside the tangent
kernel), so ``return_lse=True`` is differentiable too.

Backends implement the four array-level operations (see ``Backend``); the
"reference" backend is plain jnp and runs anywhere, the "cute" backend calls
the vendored FA4 kernels through cutejax and needs a CUDA device. All shape
canonicalization, validation, and differentiation logic above the backend is
shared — so the wiring itself is fully exercised by the CPU test suite.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jaxtyping import Array, Float

from fa4_jax.masks import MaskMod

__all__ = ["flash_attn"]


@dataclass(frozen=True)
class AttnConfig:
    """Static description of one attention variant; a primitive param, so hashable.

    ``mask`` is the mask mod's *treedef* — its class and static fields — so
    masks of the same kind with different array contents compare equal and
    share a kernel; the array leaves travel as primitive operands (``aux``)
    and ``mask.unflatten(aux)`` rebuilds the callable inside a backend.
    """

    causal: bool
    local: bool
    window: tuple[int | None, int | None]
    scale: float
    mask: jax.tree_util.PyTreeDef | None
    backend: str

    @property
    def n_aux(self) -> int:
        return 0 if self.mask is None else self.mask.num_leaves

    def mask_mod(self, aux) -> MaskMod | None:
        """Rebuild the mask callable from this config and its leaves."""
        return None if self.mask is None else self.mask.unflatten(tuple(aux))


class Backend(Protocol):
    """The four array-level ops a backend provides. All take/return jax arrays."""

    def fwd(self, cfg: AttnConfig, q, k, v, aux) -> Array: ...

    def fwd_lse(self, cfg: AttnConfig, q, k, v, aux) -> tuple[Array, Array]: ...

    def tangent(
        self, cfg: AttnConfig, q, k, v, out, lse, dq, dk, dv, aux
    ) -> tuple[Array, Array]: ...

    def vjp(
        self, cfg: AttnConfig, q, k, v, out, lse, d_out, d_lse, aux
    ) -> tuple[Array, Array, Array]: ...


_BACKENDS: dict[str, Backend] = {}


def register_backend(name: str, backend: Backend) -> None:
    _BACKENDS[name] = backend


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS and name == "cute":
        import fa4_jax.kernels  # noqa: F401  (registers itself on import)
    return _BACKENDS[name]


# Register the reference backend eagerly: it is cheap to import and is the
# fallback everywhere a GPU is missing. (Registered from here, not from the
# backend module, to keep the import one-directional.)
from fa4_jax import _reference_backend

register_backend("reference", _reference_backend.ReferenceBackend())


# ---------------------------------------------------------------- primitives

fa4_fwd_p = Primitive("fa4_fwd")
fa4_fwd_p.multiple_results = True

fa4_tangent_p = Primitive("fa4_tangent")
fa4_tangent_p.multiple_results = True


def _lse_struct(q):
    b, s_q, h, _ = q.shape
    return jax.core.ShapedArray((b, h, s_q), jnp.float32)


def _fwd_impl(q, k, v, *aux, cfg: AttnConfig, need_lse: bool):
    be = get_backend(cfg.backend)
    if need_lse:
        out, lse = be.fwd_lse(cfg, q, k, v, aux)
        return out, lse
    return (be.fwd(cfg, q, k, v, aux),)


def _fwd_abstract(q, k, v, *aux, cfg: AttnConfig, need_lse: bool):
    out = jax.core.ShapedArray((*q.shape[:3], v.shape[3]), q.dtype)
    return (out, _lse_struct(q)) if need_lse else (out,)


def _fwd_jvp(primals, tangents, *, cfg: AttnConfig, need_lse: bool):
    q, k, v, *aux = primals
    dq, dk, dv, *daux = tangents
    if any(not isinstance(t, ad.Zero) for t in daux):
        raise NotImplementedError("fa4 mask_mod array fields (aux data) are not differentiable")
    out, lse = fa4_fwd_p.bind(q, k, v, *aux, cfg=cfg, need_lse=True)
    dq, dk, dv = (
        ad.instantiate_zeros(t) if isinstance(t, ad.Zero) else t for t in (dq, dk, dv)
    )
    t_out, t_lse = fa4_tangent_p.bind(q, k, v, out, lse, dq, dk, dv, *aux, cfg=cfg)
    if need_lse:
        return (out, lse), (t_out, t_lse)
    return (out,), (t_out,)


fa4_fwd_p.def_impl(functools.partial(jax.jit, static_argnames=("cfg", "need_lse"))(_fwd_impl))
fa4_fwd_p.def_abstract_eval(_fwd_abstract)
ad.primitive_jvps[fa4_fwd_p] = _fwd_jvp
mlir.register_lowering(fa4_fwd_p, mlir.lower_fun(_fwd_impl, multiple_results=True))


def _tangent_impl(q, k, v, out, lse, dq, dk, dv, *aux, cfg: AttnConfig):
    be = get_backend(cfg.backend)
    t_out, t_lse = be.tangent(cfg, q, k, v, out, lse, dq, dk, dv, aux)
    return t_out, t_lse


def _tangent_abstract(q, k, v, out, lse, dq, dk, dv, *aux, cfg: AttnConfig):
    return jax.core.ShapedArray(out.shape, out.dtype), _lse_struct(q)


def _tangent_transpose(cts, q, k, v, out, lse, dq, dk, dv, *aux, cfg: AttnConfig):
    """Transpose of the linear-in-(dq,dk,dv) tangent map = the FA4 backward."""
    ct_out, ct_lse = cts
    assert not any(ad.is_undefined_primal(x) for x in (q, k, v, out, lse, *aux)), (
        "fa4_tangent transpose: residuals must be known primals"
    )
    ct_out = jnp.zeros(out.aval.shape, out.aval.dtype) if isinstance(ct_out, ad.Zero) else ct_out
    d_lse = None if isinstance(ct_lse, ad.Zero) else ct_lse
    be = get_backend(cfg.backend)
    dq_bar, dk_bar, dv_bar = be.vjp(cfg, q, k, v, out, lse, ct_out, d_lse, aux)
    return (None, None, None, None, None, dq_bar, dk_bar, dv_bar, *[None] * len(aux))


fa4_tangent_p.def_impl(functools.partial(jax.jit, static_argnames=("cfg",))(_tangent_impl))
fa4_tangent_p.def_abstract_eval(_tangent_abstract)
ad.primitive_transposes[fa4_tangent_p] = _tangent_transpose
mlir.register_lowering(fa4_tangent_p, mlir.lower_fun(_tangent_impl, multiple_results=True))


# ---------------------------------------------------------------- batching
#
# Every operand and result of both primitives carries batch as its leading
# axis, so vmap folds into it: move the mapped axis to the front, broadcast
# unmapped operands across it, reshape (vmap, batch, ...) -> (vmap*batch, ...)
# and rebind the same primitive. That keeps the custom jvp/transpose rules in
# the graph (jacfwd, vmap-of-grad, ...) and maps onto the kernels' batch
# parallelism directly.
#
# A mask_mod (or its array leaves) can depend on the batch index, which
# folding would renumber; those fall back to vmapping the impl instead —
# semantically right, but it differentiates through the backend's jnp code
# rather than the primitives, so on the cute backend it is unsupported.


def _batch_via_fold(prim, args, dims, **params):
    size = next(a.shape[d] for a, d in zip(args, dims) if d is not None)
    folded = []
    for a, d in zip(args, dims):
        if d is None:
            a = jnp.broadcast_to(a[None], (size, *a.shape))
        else:
            a = jnp.moveaxis(a, d, 0)
        folded.append(a.reshape(size * a.shape[1], *a.shape[2:]))
    res = prim.bind(*folded, **params)
    return tuple(r.reshape(size, -1, *r.shape[1:]) for r in res), (0,) * len(res)


def _batch_via_impl(impl, args, dims, **params):
    f = jax.vmap(functools.partial(impl, **params), in_axes=tuple(dims), out_axes=0)
    res = f(*args)
    return res, (0,) * len(res)


def _fwd_batch(args, dims, *, cfg: AttnConfig, need_lse: bool):
    if cfg.mask is None:
        return _batch_via_fold(fa4_fwd_p, args, dims, cfg=cfg, need_lse=need_lse)
    return _batch_via_impl(_fwd_impl, args, dims, cfg=cfg, need_lse=need_lse)


def _tangent_batch(args, dims, *, cfg: AttnConfig):
    if cfg.mask is None:
        return _batch_via_fold(fa4_tangent_p, args, dims, cfg=cfg)
    return _batch_via_impl(_tangent_impl, args, dims, cfg=cfg)


batching.primitive_batchers[fa4_fwd_p] = _fwd_batch
batching.primitive_batchers[fa4_tangent_p] = _tangent_batch


# ---------------------------------------------------------------- public API


def _resolve_causal_local_window(causal, left, right):
    """Canonicalise (causal, window) exactly as flash_attn.cute.interface does.

    ``causal`` and ``local`` end up mutually exclusive: a bare right-window of
    0 is the causal kernel, anything else with a bound is the local kernel.
    """
    if causal:
        right = 0
    if left is not None and right is not None and left + right < 0:
        left = right = None
    if left is None and right is None:
        return causal, False, None, None
    if left is None and right == 0:
        return True, False, None, None
    return False, True, left, right


def _default_backend() -> str:
    return "cute" if jax.default_backend() == "gpu" else "reference"


def flash_attn(
    q: Float[Array, "batch q_seq q_heads head_dim"],
    k: Float[Array, "batch kv_seq kv_heads head_dim"],
    v: Float[Array, "batch kv_seq kv_heads head_dim_v"],
    *,
    causal: bool = False,
    window_size: tuple[int | None, int | None] | None = None,
    scale: float | None = None,
    mask_mod: MaskMod | None = None,
    return_lse: bool = False,
    backend: str | None = None,
):
    """FlashAttention-4 with reverse- and forward-mode differentiation.

    Args:
        q, k, v: ``(batch, seqlen, heads, head_dim)``; fp16/bf16 on the cute
            backend, any float on the reference backend. GQA by giving k/v
            fewer heads (must divide q's).
        causal: mask out ``kv_idx > q_idx``; equivalent to
            ``window_size=(None, 0)`` and takes the kernel's causal path.
        window_size: ``(left, right)`` sliding window in keys, counted from
            the query position and inclusive of the band
            ``-right <= q - kv <= left``; ``None`` on either side means
            unbounded. Combines with ``causal``. Out-of-band KV blocks are
            *skipped*, not masked.
        scale: softmax scale, default ``1/sqrt(head_dim)``.
        mask_mod: a :class:`fa4_jax.MaskMod` flex-attention predicate — a
            callable pytree. Its class and static fields are baked into the
            kernel as code (one compiled kernel per mask *kind*); its array
            fields are runtime data, so new contents reuse the kernel. Not
            differentiable through those arrays. Combines (AND) with
            ``causal``/``window_size``.
        return_lse: also return the row log-sum-exp ``(batch, heads, q_seq)``
            (float32, natural log; ``-inf`` for rows with no visible key).
        backend: ``"cute"`` (FA4 kernels, CUDA), ``"reference"`` (plain jnp),
            or None to pick cute exactly when jax is running on GPU.

    Returns:
        ``(batch, q_seq, q_heads, head_dim_v)`` in q's dtype, plus the LSE if
        ``return_lse``. Rows with no visible key return zeros.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("flash_attn takes (batch, seqlen, heads, head_dim) arrays")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q/k/v dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError(f"k/v seqlen+heads must match, got {k.shape} vs {v.shape}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"batch mismatch: q {q.shape[0]} vs kv {k.shape[0]}")
    if q.shape[3] != k.shape[3]:
        raise ValueError(f"q/k head_dim mismatch: {q.shape[3]} vs {k.shape[3]}")
    if q.shape[2] % k.shape[2]:
        raise ValueError(f"q heads {q.shape[2]} not a multiple of kv heads {k.shape[2]}")
    if mask_mod is not None and not isinstance(mask_mod, MaskMod):
        raise TypeError(
            "mask_mod must be a fa4_jax.MaskMod (a raw callable can't run on "
            "the reference backend; wrap the predicate in fa4_jax.FnMask)"
        )
    aux_tensors, mask_def = jax.tree.flatten(mask_mod)
    for leaf in aux_tensors:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise TypeError(
                f"mask_mod field holds {leaf!r}, which is not an array. Non-array "
                "fields must be declared static (fa4_jax.masks.static()) so they "
                "live in the treedef rather than becoming traced operands."
            )
    if mask_mod is None:
        mask_def = None

    left, right = window_size if window_size is not None else (None, None)
    causal, local, wl, wr = _resolve_causal_local_window(
        causal, None if left is None else int(left), None if right is None else int(right)
    )
    cfg = AttnConfig(
        causal=causal,
        local=local,
        window=(wl, wr),
        scale=float(scale if scale is not None else 1.0 / math.sqrt(q.shape[3])),
        mask=mask_def,
        backend=backend if backend is not None else _default_backend(),
    )
    res = fa4_fwd_p.bind(q, k, v, *aux_tensors, cfg=cfg, need_lse=return_lse)
    return (res[0], res[1]) if return_lse else res[0]
