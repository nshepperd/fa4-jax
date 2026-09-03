"""Public ``flash_attn`` and its differentiation wiring.

The function is built on three JAX primitives rather than ``jax.custom_vjp``,
because it supports *both* differentiation modes and custom_vjp forbids
forward mode:

- ``fa4_fwd``: ``(q, k, v, *aux) -> (out[, lse])``. Its JVP rule computes the
  primal via the kernel forward (with LSE) and the tangent via ``fa4_tangent``.
- ``fa4_tangent``: ``(q, k, v, out, lse, dq, dk, dv, *aux) -> (t_out, t_lse)``.
  Linear in ``(dq, dk, dv)``; its *transpose* rule binds
- ``fa4_vjp``: ``(q, k, v, out, lse, d_out[, d_lse], *aux) -> (dq, dk, dv)``,
  the FA4 backward pass (a primitive of its own so it batches under vmap).

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
shared — so the wiring is checked once, against plain JAX autodiff of the
reference, and the kernels are checked against the reference.

``flash_attn_varlen`` shares the same three primitives: ``cfg.varlen``
switches the operand layout to packed ``(total_tokens, heads, head_dim)``
q/k/v followed by four int32 seqlen operands — cu_seqlens_q, cu_seqlens_k,
seqused_q, seqused_k, the last two synthesized as the slot sizes when the
caller does not pass them, so the operand count stays fixed — which
differentiation treats as data: integer tangents are symbolic zeros,
transposes return no cotangent for them.
"""

from __future__ import annotations

import functools
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jaxtyping import Array, Float

from fa4_jax.masks import MaskMod

__all__ = ["default_kernel_arch", "flash_attn", "flash_attn_varlen"]


# ---------------------------------------------------------------- kernel families

# Ambient default for the per-call `kernel_arch` argument (explicit argument
# wins). Process-global: meant for tests and benchmarks that want every call
# in a region to use one kernel family without threading the parameter.
_DEFAULT_KERNEL_ARCH: int | None = None


@contextmanager
def default_kernel_arch(arch: int | None):
    """Scope a default for ``kernel_arch``: every ``flash_attn`` /
    ``flash_attn_varlen`` call inside behaves as if ``kernel_arch=arch`` was
    passed (calls that pass it explicitly are unaffected). Not thread-safe.
    """
    global _DEFAULT_KERNEL_ARCH
    prev = _DEFAULT_KERNEL_ARCH
    _DEFAULT_KERNEL_ARCH = None if arch is None else int(arch)
    try:
        yield
    finally:
        _DEFAULT_KERNEL_ARCH = prev


def kernel_arch_unsupported(arch: int, device_arch: int) -> str | None:
    """Why kernel family ``arch`` cannot execute on ``sm_{device_arch}``, or
    None if it can. The single source of truth for the runtime validation in
    the cute backend and the test suite's per-family skip logic.

    A family is the *implementation* (instruction mix, schedules, tiles);
    code is always compiled for the real device, so a family can serve
    newer hardware when its instructions exist there: the SM80-MMA kernels
    (cp.async + mma.sync, and tile configs that fit the smallest 99 KB SMEM
    budget) run on sm_8x, sm_90, sm_10x/11x, and sm_12x alike; the wgmma
    (Hopper) kernels need sm_90 -- wgmma and its setmaxnreg warp
    specialization exist on no other generation; the tcgen05 kernels need
    datacenter Blackwell.
    """
    if arch // 10 == 9:
        if device_arch // 10 != 9:
            return "the SM90 (wgmma) kernels need a Hopper (sm_90) device"
        return None
    if arch // 10 in (10, 11):
        if device_arch // 10 not in (10, 11):
            return "the tcgen05 kernels need a datacenter-Blackwell (sm_10x) device"
        return None
    if arch // 10 in (8, 12):
        if device_arch // 10 not in (8, 9, 10, 11, 12):
            return f"the SM80-MMA kernel family is not supported on sm_{device_arch}"
        return None
    return "unknown kernel family (expected one of 8x, 90, 10x, 12x)"


@dataclass(frozen=True)
class AttnConfig:
    """Static description of one attention variant; a primitive param, so hashable.

    ``mask`` is the mask mod's *treedef* — its class and static fields — so
    masks of the same kind with different array contents compare equal and
    share a kernel; the array leaves travel as primitive operands (``aux``)
    and ``mask.unflatten(aux)`` rebuilds the callable inside a backend.

    ``varlen`` switches the primitives to the packed layout: q/k/v are
    ``(total_tokens, heads, head_dim)`` and four extra int32 seqlen operands
    (cu_seqlens_q/k cumulative offsets, seqused_q/k used lengths) follow
    q/k/v in the operand lists.
    """

    causal: bool
    local: bool
    window: tuple[int | None, int | None]
    scale: float
    mask: jax.tree_util.PyTreeDef | None
    backend: str
    varlen: bool = False
    # Kernel-family override for the cute backend (e.g. 80 to run the
    # SM80-MMA kernels on newer hardware); None = follow the device.
    kernel_arch: int | None = None
    # Varlen only: give unused rows (beyond seqused, or in cu_seqlens gaps)
    # defined values — 0 in out/tangent/gradients, -inf in the LSE — instead
    # of unspecified memory. Costs one buffer fill per output on the cute
    # backend; the reference backend behaves this way regardless.
    defined_padding: bool = False

    @property
    def n_aux(self) -> int:
        return 0 if self.mask is None else self.mask.num_leaves

    def mask_mod(self, aux) -> MaskMod | None:
        """Rebuild the mask callable from this config and its leaves."""
        return None if self.mask is None else self.mask.unflatten(tuple(aux))


class Backend(Protocol):
    """The four array-level ops a backend provides. All take/return jax arrays.

    ``seqlens`` is ``None`` for the batched layout, or the varlen 4-tuple
    ``(cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)`` — all always
    present in varlen mode (seqused defaults to the slot sizes).
    """

    def fwd(self, cfg: AttnConfig, q, k, v, aux, seqlens=None) -> Array: ...

    def fwd_lse(
        self, cfg: AttnConfig, q, k, v, aux, seqlens=None
    ) -> tuple[Array, Array]: ...

    def tangent(
        self, cfg: AttnConfig, q, k, v, out, lse, dq, dk, dv, aux, seqlens=None
    ) -> tuple[Array, Array]: ...

    def vjp(
        self, cfg: AttnConfig, q, k, v, out, lse, d_out, d_lse, aux, seqlens=None
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


def _lse_struct(q, cfg: AttnConfig):
    if cfg.varlen:
        total_q, h, _ = q.shape
        return jax.core.ShapedArray((h, total_q), jnp.float32)
    b, s_q, h, _ = q.shape
    return jax.core.ShapedArray((b, h, s_q), jnp.float32)


def _fwd_split(cfg: AttnConfig, args):
    """``fa4_fwd`` operands -> (q, k, v, seqlens, aux); seqlens is the
    (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k) 4-tuple when
    ``cfg.varlen`` else None."""
    if cfg.varlen:
        q, k, v, cu_q, cu_k, su_q, su_k, *aux = args
        return q, k, v, (cu_q, cu_k, su_q, su_k), tuple(aux)
    q, k, v, *aux = args
    return q, k, v, None, tuple(aux)


def _fwd_impl(*args, cfg: AttnConfig, need_lse: bool):
    q, k, v, seqlens, aux = _fwd_split(cfg, args)
    be = get_backend(cfg.backend)
    if need_lse:
        out, lse = be.fwd_lse(cfg, q, k, v, aux, seqlens)
        return out, lse
    return (be.fwd(cfg, q, k, v, aux, seqlens),)


def _fwd_abstract(*args, cfg: AttnConfig, need_lse: bool):
    q, _k, v, _, _ = _fwd_split(cfg, args)
    out = jax.core.ShapedArray((*q.shape[:-1], v.shape[-1]), q.dtype)
    return (out, _lse_struct(q, cfg)) if need_lse else (out,)


def _fwd_jvp(primals, tangents, *, cfg: AttnConfig, need_lse: bool):
    q, k, v, seqlens, aux = _fwd_split(cfg, primals)
    dq, dk, dv, _, daux = _fwd_split(cfg, tangents)
    if any(not isinstance(t, ad.Zero) for t in daux):
        raise NotImplementedError(
            "fa4 mask_mod array fields (aux data) are not differentiable"
        )
    out, lse = fa4_fwd_p.bind(*primals, cfg=cfg, need_lse=True)
    dq, dk, dv = (
        ad.instantiate_zeros(t) if isinstance(t, ad.Zero) else t for t in (dq, dk, dv)
    )
    t_out, t_lse = fa4_tangent_p.bind(
        q, k, v, out, lse, dq, dk, dv, *(seqlens or ()), *aux, cfg=cfg
    )
    if need_lse:
        return (out, lse), (t_out, t_lse)
    return (out,), (t_out,)


fa4_fwd_p.def_impl(
    functools.partial(jax.jit, static_argnames=("cfg", "need_lse"))(_fwd_impl)
)
fa4_fwd_p.def_abstract_eval(_fwd_abstract)
ad.primitive_jvps[fa4_fwd_p] = _fwd_jvp
mlir.register_lowering(fa4_fwd_p, mlir.lower_fun(_fwd_impl, multiple_results=True))


def _tangent_split(cfg: AttnConfig, args):
    q, k, v, out, lse, dq, dk, dv, *rest = args
    if cfg.varlen:
        cu_q, cu_k, su_q, su_k, *aux = rest
        return q, k, v, out, lse, dq, dk, dv, (cu_q, cu_k, su_q, su_k), tuple(aux)
    return q, k, v, out, lse, dq, dk, dv, None, tuple(rest)


def _tangent_impl(*args, cfg: AttnConfig):
    q, k, v, out, lse, dq, dk, dv, seqlens, aux = _tangent_split(cfg, args)
    be = get_backend(cfg.backend)
    t_out, t_lse = be.tangent(cfg, q, k, v, out, lse, dq, dk, dv, aux, seqlens)
    return t_out, t_lse


def _tangent_abstract(*args, cfg: AttnConfig):
    q, _, _, out, *_ = args
    return jax.core.ShapedArray(out.shape, out.dtype), _lse_struct(q, cfg)


def _tangent_transpose(cts, *args, cfg: AttnConfig):
    """Transpose of the linear-in-(dq,dk,dv) tangent map = the FA4 backward."""
    q, k, v, out, lse, _dq, _dk, _dv, seqlens, aux = _tangent_split(cfg, args)
    ct_out, ct_lse = cts
    assert not any(
        ad.is_undefined_primal(x) for x in (q, k, v, out, lse, *(seqlens or ()), *aux)
    ), "fa4_tangent transpose: residuals must be known primals"
    ct_out = (
        jnp.zeros(out.aval.shape, out.aval.dtype)
        if isinstance(ct_out, ad.Zero)
        else ct_out
    )
    d_lse = () if isinstance(ct_lse, ad.Zero) else (ct_lse,)
    dq_bar, dk_bar, dv_bar = fa4_vjp_p.bind(
        q, k, v, out, lse, ct_out, *d_lse, *(seqlens or ()), *aux,
        cfg=cfg, has_d_lse=bool(d_lse),
    )
    n_tail = len(seqlens or ()) + len(aux)
    return (None, None, None, None, None, dq_bar, dk_bar, dv_bar, *[None] * n_tail)


fa4_tangent_p.def_impl(
    functools.partial(jax.jit, static_argnames=("cfg",))(_tangent_impl)
)
fa4_tangent_p.def_abstract_eval(_tangent_abstract)
ad.primitive_transposes[fa4_tangent_p] = _tangent_transpose
mlir.register_lowering(
    fa4_tangent_p, mlir.lower_fun(_tangent_impl, multiple_results=True)
)


# The backward is its own primitive (rather than a direct backend call inside
# the transpose rule) so that it, too, has a batching rule: the transpose
# runs under any enclosing vmap (per-example gradients), and a raw cutejax
# call there would have no way to batch.

fa4_vjp_p = Primitive("fa4_vjp")
fa4_vjp_p.multiple_results = True


def _vjp_split(cfg, args, has_d_lse):
    q, k, v, out, lse, d_out, *rest = args
    d_lse, rest = (rest[0], rest[1:]) if has_d_lse else (None, rest)
    if cfg.varlen:
        cu_q, cu_k, su_q, su_k, *aux = rest
        return q, k, v, out, lse, d_out, d_lse, (cu_q, cu_k, su_q, su_k), tuple(aux)
    return q, k, v, out, lse, d_out, d_lse, None, tuple(rest)


def _vjp_impl(*args, cfg: AttnConfig, has_d_lse: bool):
    q, k, v, out, lse, d_out, d_lse, seqlens, aux = _vjp_split(cfg, args, has_d_lse)
    be = get_backend(cfg.backend)
    return tuple(be.vjp(cfg, q, k, v, out, lse, d_out, d_lse, aux, seqlens))


def _vjp_abstract(*args, cfg: AttnConfig, has_d_lse: bool):
    q, k, v, *_ = args
    return tuple(jax.core.ShapedArray(x.shape, x.dtype) for x in (q, k, v))


fa4_vjp_p.def_impl(
    functools.partial(jax.jit, static_argnames=("cfg", "has_d_lse"))(_vjp_impl)
)
fa4_vjp_p.def_abstract_eval(_vjp_abstract)
mlir.register_lowering(fa4_vjp_p, mlir.lower_fun(_vjp_impl, multiple_results=True))


# ---------------------------------------------------------------- batching
#
# Every operand and result of the three primitives carries batch as its leading
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


def _no_varlen_vmap(cfg: AttnConfig):
    if cfg.varlen:
        raise NotImplementedError(
            "vmap over flash_attn_varlen is not supported: the packed layout "
            "already encodes its batch in cu_seqlens"
        )


def _fwd_batch(args, dims, *, cfg: AttnConfig, need_lse: bool):
    _no_varlen_vmap(cfg)
    if cfg.mask is None:
        return _batch_via_fold(fa4_fwd_p, args, dims, cfg=cfg, need_lse=need_lse)
    return _batch_via_impl(_fwd_impl, args, dims, cfg=cfg, need_lse=need_lse)


def _tangent_batch(args, dims, *, cfg: AttnConfig):
    _no_varlen_vmap(cfg)
    if cfg.mask is None:
        return _batch_via_fold(fa4_tangent_p, args, dims, cfg=cfg)
    return _batch_via_impl(_tangent_impl, args, dims, cfg=cfg)


def _vjp_batch(args, dims, *, cfg: AttnConfig, has_d_lse: bool):
    _no_varlen_vmap(cfg)
    if cfg.mask is None:
        return _batch_via_fold(fa4_vjp_p, args, dims, cfg=cfg, has_d_lse=has_d_lse)
    return _batch_via_impl(_vjp_impl, args, dims, cfg=cfg, has_d_lse=has_d_lse)


batching.primitive_batchers[fa4_fwd_p] = _fwd_batch
batching.primitive_batchers[fa4_tangent_p] = _tangent_batch
batching.primitive_batchers[fa4_vjp_p] = _vjp_batch


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


def _flatten_mask(mask_mod):
    """Validate a mask_mod and split it into (aux array leaves, treedef)."""
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
    return aux_tensors, (None if mask_mod is None else mask_def)


def _check_qkv(q, k, v):
    """Shape/dtype agreement checks shared by both layouts; seqlen and heads
    are the two axes before head_dim, batch (if any) leads."""
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            f"q/k/v dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}"
        )
    if k.shape[-3] != v.shape[-3] or k.shape[-2] != v.shape[-2]:
        raise ValueError(f"k/v seqlen+heads must match, got {k.shape} vs {v.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k head_dim mismatch: {q.shape[-1]} vs {k.shape[-1]}")
    if q.shape[-2] % k.shape[-2]:
        raise ValueError(
            f"q heads {q.shape[-2]} not a multiple of kv heads {k.shape[-2]}"
        )


def _make_config(
    q, causal, window_size, scale, mask_def, backend, varlen, kernel_arch,
    defined_padding=False,
):
    left, right = window_size if window_size is not None else (None, None)
    causal, local, wl, wr = _resolve_causal_local_window(
        causal,
        None if left is None else int(left),
        None if right is None else int(right),
    )
    if kernel_arch is None:
        kernel_arch = _DEFAULT_KERNEL_ARCH
    return AttnConfig(
        causal=causal,
        local=local,
        window=(wl, wr),
        scale=float(scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])),
        mask=mask_def,
        backend=backend if backend is not None else _default_backend(),
        varlen=varlen,
        kernel_arch=None if kernel_arch is None else int(kernel_arch),
        defined_padding=bool(defined_padding),
    )


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
    kernel_arch: int | None = None,
):
    """FlashAttention-4 with reverse- and forward-mode differentiation.

    Args:
        q, k, v: ``(batch, seqlen, heads, head_dim)``; fp16/bf16 on the cute
            backend, any float on the reference backend. GQA by giving k/v
            fewer heads (must divide q's).
        causal: mask out keys after the query's position; equivalent to
            ``window_size=(None, 0)`` and takes the kernel's causal path.
        window_size: ``(left, right)`` sliding window in keys, counted from
            the query position and inclusive of the band
            ``-right <= q_pos - kv <= left``; ``None`` on either side means
            unbounded. Combines with ``causal``. Out-of-band KV blocks are
            *skipped*, not masked.

            Both are **bottom-right aligned** (the flash-attention
            convention): ``q_pos = q_idx + (kv_seq - q_seq)``, so with
            ``q_seq != kv_seq`` the last query lines up with the last key —
            the decode / chunked-prefill case. ``mask_mod`` predicates see
            raw ``q_idx``/``kv_idx`` and are unaffected, so
            ``mask_mod=causal_mask()`` is top-left and differs from
            ``causal=True`` unless ``q_seq == kv_seq``.
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
        kernel_arch: override which kernel *family* the cute backend uses,
            as a compute capability (e.g. ``80`` for the SM80-MMA kernels,
            ``120`` for their SM120 variant, ``90`` for the Hopper wgmma
            kernels, ``100`` for tcgen05); ``None`` follows the ambient
            :func:`default_kernel_arch` scope if one is active, else the
            device. Code is still compiled for the real device — this only
            selects the implementation, which must be executable there (the
            SM80/SM120 families run on sm_8x/sm_90/sm_100/sm_120; wgmma
            needs sm_90; tcgen05 needs sm_100). The reference backend
            ignores it. Used by the test suite to cover every runnable
            family; also an escape hatch if one family misbehaves.

    Returns:
        ``(batch, q_seq, q_heads, head_dim_v)`` in q's dtype, plus the LSE if
        ``return_lse``. Rows with no visible key return zeros.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("flash_attn takes (batch, seqlen, heads, head_dim) arrays")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"batch mismatch: q {q.shape[0]} vs kv {k.shape[0]}")
    _check_qkv(q, k, v)
    aux_tensors, mask_def = _flatten_mask(mask_mod)
    cfg = _make_config(q, causal, window_size, scale, mask_def, backend, varlen=False, kernel_arch=kernel_arch)
    res = fa4_fwd_p.bind(q, k, v, *aux_tensors, cfg=cfg, need_lse=return_lse)
    return (res[0], res[1]) if return_lse else res[0]


def flash_attn_varlen(
    q: Float[Array, "total_q q_heads head_dim"],
    k: Float[Array, "total_k kv_heads head_dim"],
    v: Float[Array, "total_k kv_heads head_dim_v"],
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    seqused_q=None,
    seqused_k=None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] | None = None,
    scale: float | None = None,
    mask_mod: MaskMod | None = None,
    return_lse: bool = False,
    backend: str | None = None,
    kernel_arch: int | None = None,
    defined_padding: bool = False,
):
    """Variable-length (packed / ragged) FlashAttention-4.

    A batch of sequences of different lengths is packed by concatenation
    along the token axis — no padding — and delimited by cumulative-offset
    arrays, the flash-attention ``varlen`` convention:

    Args:
        q: ``(total_q, q_heads, head_dim)``, all sequences concatenated.
        k, v: ``(total_k, kv_heads, head_dim[/ _v])``; GQA by giving k/v
            fewer heads (must divide q's). The K/V packing may differ from
            q's (e.g. one query per sequence against long caches).
        cu_seqlens_q: int32 ``(batch + 1,)`` cumulative sequence starts:
            sequence ``b`` is ``q[cu_seqlens_q[b] : cu_seqlens_q[b + 1]]``.
            Must be nondecreasing with ``cu_seqlens_q[0] == 0`` and
            ``cu_seqlens_q[-1] == total_q`` (contents are runtime data and
            are **not checked**; tokens not covered by any sequence leave
            unspecified values in the outputs and their gradients).
            Zero-length sequences (repeated offsets) are fine.
        cu_seqlens_k: same for k/v; must have the same ``batch``.
        seqused_q, seqused_k: optional int32 ``(batch,)`` *used* lengths.
            When given, only the first ``seqused[b]`` tokens of slot ``b``
            participate: on the K side the rest of the slot is invisible (the
            KV-cache case — slots allocated larger than currently filled); on
            the Q side the rest produces no output (those rows of the result
            and its gradients are unspecified, like uncovered tokens).
            ``causal``/``window_size`` align on the *used* lengths. Must
            satisfy ``0 <= seqused[b] <= cu[b+1] - cu[b]`` (runtime data, not
            checked). Default: the full slot, ``cu[1:] - cu[:-1]``.
        causal, window_size: as in :func:`flash_attn`, applied **per
            sequence** and bottom-right aligned within it: query ``i`` of
            sequence ``b`` sits at key position ``i + (len_k_b - len_q_b)``.
        scale: softmax scale, default ``1/sqrt(head_dim)``.
        mask_mod: as in :func:`flash_attn`; the predicate sees the *sequence*
            index as ``b`` and **sequence-local** ``q_idx``/``kv_idx`` (both
            backends agree). Attention never crosses sequences regardless.
        return_lse: also return the row log-sum-exp, laid out
            ``(q_heads, total_q)`` (float32, natural log, ``-inf`` for rows
            with no visible key).
        backend: as in :func:`flash_attn`.
        kernel_arch: as in :func:`flash_attn`.
        defined_padding: unused rows — beyond ``seqused``, or tokens not
            covered by any sequence — are normally *unspecified* in the
            outputs and their gradients. With ``True`` they are defined:
            0 in the output, its tangent, and dq/dk/dv, and ``-inf`` in the
            LSE (the reference backend's convention, so the backends agree
            on the full buffer). Costs one buffer fill per output on the
            cute backend.

    Returns:
        ``(total_q, q_heads, head_dim_v)`` in q's dtype, plus the LSE if
        ``return_lse``.

    Differentiation: reverse mode runs the FA4 varlen backward kernels and
    forward mode (``jax.jvp``) the fused varlen tangent kernel. ``vmap``
    over a varlen call is not supported (the packing *is* the batch).
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "flash_attn_varlen takes packed (total_tokens, heads, head_dim) arrays"
        )
    _check_qkv(q, k, v)
    cu_seqlens_q, cu_seqlens_k = (
        jnp.asarray(cu) for cu in (cu_seqlens_q, cu_seqlens_k)
    )
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if cu.ndim != 1 or cu.shape[0] < 2:
            raise ValueError(
                f"{name} must be a 1D (batch + 1,) array with batch >= 1, "
                f"got shape {cu.shape}"
            )
        if not jnp.issubdtype(cu.dtype, jnp.integer):
            raise TypeError(f"{name} must be an integer array, got {cu.dtype}")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError(
            f"cu_seqlens_q/k batch mismatch: {cu_seqlens_q.shape} vs "
            f"{cu_seqlens_k.shape}"
        )
    cu_seqlens_q = cu_seqlens_q.astype(jnp.int32)
    cu_seqlens_k = cu_seqlens_k.astype(jnp.int32)
    batch = cu_seqlens_q.shape[0] - 1
    # The primitives always carry both seqused operands so the operand count
    # (and the compiled kernel variant) is fixed: a full slot is the same as
    # no override.
    seqused = []
    for name, su, cu in (
        ("seqused_q", seqused_q, cu_seqlens_q),
        ("seqused_k", seqused_k, cu_seqlens_k),
    ):
        if su is None:
            seqused.append(cu[1:] - cu[:-1])
            continue
        su = jnp.asarray(su)
        if su.ndim != 1 or su.shape[0] != batch:
            raise ValueError(
                f"{name} must be a 1D (batch,) array matching cu_seqlens "
                f"(batch={batch}), got shape {su.shape}"
            )
        if not jnp.issubdtype(su.dtype, jnp.integer):
            raise TypeError(f"{name} must be an integer array, got {su.dtype}")
        seqused.append(su.astype(jnp.int32))
    aux_tensors, mask_def = _flatten_mask(mask_mod)
    cfg = _make_config(
        q, causal, window_size, scale, mask_def, backend, varlen=True,
        kernel_arch=kernel_arch, defined_padding=defined_padding,
    )
    res = fa4_fwd_p.bind(
        q, k, v, cu_seqlens_q, cu_seqlens_k, *seqused, *aux_tensors,
        cfg=cfg, need_lse=return_lse,
    )
    return (res[0], res[1]) if return_lse else res[0]
