"""The cute backend: vendored FA4 kernels driven through cutejax.

A port of minimax-music3's fa4.py restructured behind the Backend protocol,
plus flex-attention mask mods: the MaskMod predicate is wrapped into a
``@cute.jit`` callable (scalar Int32 indices in, Boolean SSA out) and handed
to the kernel constructor, so it becomes codegen; the kernel-config dict rides
the cutejax pytree as an identity-hashed static leaf, making each distinct
MaskMod object its own compile-cache entry while aux tensors stay runtime
data.

Kernel-side restrictions surface here as NotImplementedError:

- backward (reverse mode): no GQA (upstream's dK/dV accumulation assumes
  qhead_per_kvhead == 1 here), no mask_mod (upstream SM80 backward predates
  it), no lse cotangent. The JVP tangent kernel supports all three.
- fp16/bf16 only, head_dim % 8 == 0 (validated with clear errors).

Layout note (from fa4.py): heads and head_dim are baked into the kernel (they
pick tile sizes and layouts); batch and seqlen stay symbolic, so one compile
serves every length.
"""

from __future__ import annotations

import functools

import cutejax
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute

from fa4_jax.attention import AttnConfig, register_backend
from fa4_jax.masks import MaskMod

_QKV_SPEC = cutejax.ArraySpec(static_dims=(2, 3))

_DTYPES = {jnp.dtype(jnp.bfloat16): cutlass.BFloat16, jnp.dtype(jnp.float16): cutlass.Float16}

# The backward's fp32 scratch tensors have no static dims, so the compiler
# only vectorizes their copies if we promise the innermost dim is a multiple
# of 4 (4 x f32 = the 128-bit copy atom). Padded to a tile boundary, they are.
# LSE is the exception: its length is the raw seqlen, so no promise is made.
_F32_SPEC = cutejax.ArraySpec(divisibility=4)
_LSE_SPEC = cutejax.ArraySpec()


def _device_arch() -> int:
    """Compute capability of the default device as an int, e.g. 86 or 120."""
    cap = getattr(jax.devices()[0], "compute_capability", None)
    if cap is None:
        raise RuntimeError("the fa4-jax cute backend needs a CUDA device; jax is not on GPU")
    major, _, minor = str(cap).partition(".")
    return int(major) * 10 + int(minor or 0)


@functools.cache
def _cute_mask_mod(mask: MaskMod):
    """Stage a MaskMod predicate as the cute callable the kernels expect.

    The kernels hand the predicate size-1 SSA vectors; the MaskMod contract is
    scalar indices and a scalar Boolean, so scalarize on the way in and re-wrap
    on the way out. Cached per MaskMod identity: the returned function object
    is baked into kernel configs and must be stable.
    """
    from fa4_jax.cute.utils import scalar_to_ssa

    @cute.jit
    def mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        pred = mask.fn(batch_idx[0], head_idx[0], q_idx[0], kv_idx[0], aux_tensors)
        return scalar_to_ssa(pred, cutlass.Boolean)

    return mod


# ---------------------------------------------------------------- forward


@functools.cache
def _fwd_kernel_cls(arch: int):
    # SM120 differs from SM80 only in the shared-memory capacity check, but
    # that check is what keeps 128x128 tiles from silently overflowing 99 KB.
    if arch // 10 == 12:
        from fa4_jax.cute.flash_fwd_sm120 import FlashAttentionForwardSm120

        return FlashAttentionForwardSm120
    if arch // 10 == 8:
        from fa4_jax.cute.flash_fwd import FlashAttentionForwardSm80

        return FlashAttentionForwardSm80
    raise NotImplementedError(
        f"fa4-jax wraps the SM80/SM120 kernels only; device is sm_{arch}. "
        "Hopper (sm_90) and datacenter Blackwell (sm_100) have their own FA4 "
        "kernels that are not vendored here."
    )


def _fwd_tile_mn(arch: int, head_dim: int) -> tuple[int, int]:
    """Tile sizes matching flash_attn.cute.interface's per-arch heuristic."""
    if arch // 10 == 12 and head_dim <= 64:
        return 128, 128  # 48 KB of the 99 KB SMEM budget
    return 128, 64


@functools.cache
def _get_fwd(dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, mask_mod, n_aux, arch):
    tile_m, tile_n = _fwd_tile_mn(arch, head_dim)
    return _fwd_kernel_cls(arch)(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        is_causal=causal,
        is_local=local,
        pack_gqa=False,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=1,
        num_threads=128,
        Q_in_regs=False,
        mask_mod=None if mask_mod is None else _cute_mask_mod(mask_mod),
        has_aux_tensors=n_aux > 0,
    )


def _windows(cfg):
    """Window bounds as the kernels want them.

    ``cfg`` is a static pytree leaf, so these are plain python ints at trace
    time; they still need an explicit cast, because the kernels annotate them
    ``Int32 | None`` and we are already inside a cute trace, where the DSL's
    host-boundary int->Int32 coercion does not apply.
    """
    return tuple(None if w is None else cutlass.Int32(w) for w in (cfg["wl"], cfg["wr"]))


def _aux_data(tree):
    from fa4_jax.cute.utils import AuxData

    n = tree["n_aux"]
    if n == 0:
        return AuxData()
    return AuxData(tensors=tuple(tree[f"aux{i}"] for i in range(n)))


@cute.jit
def _fa4_fwd(stream, tree, mO):
    wl, wr = _windows(tree)
    tree["fa"](
        tree["q"], tree["k"], tree["v"], mO, None, tree["scale"],
        window_size_left=wl, window_size_right=wr,
        aux_data=_aux_data(tree), stream=stream,
    )


@cute.jit
def _fa4_fwd_lse(stream, tree, mO, mLSE):
    """Same kernel, but also writing the log-sum-exp the tangent/backward need."""
    wl, wr = _windows(tree)
    tree["fa"](
        tree["q"], tree["k"], tree["v"], mO, mLSE, tree["scale"],
        window_size_left=wl, window_size_right=wr,
        aux_data=_aux_data(tree), stream=stream,
    )


# ---------------------------------------------------------------- backward
#
# FA4 splits the backward into three kernels, and so do we:
#
#   preprocess  ->  dpsum = (o * do).sum(-1), lse_log2 = lse * log2(e),
#                   and a zeroed fp32 dQ accumulator
#   backward    ->  dK, dV directly; dQ atomically accumulated into that fp32
#                   buffer (aliased in place, hence `input_output_aliases`)
#   postprocess ->  dQaccum -> dQ in the input dtype, scaled by softmax_scale
#
# Tile config: upstream has no sm_8x branch for the backward -- arch 8 falls
# through to the SM100 settings, which ask for 128x128 tiles that will not fit
# sm_86's 100 KB of SMEM. We use the SM120 config (64x64, 128 threads) on both
# arches instead: conservative, fits everywhere, and it is the only one
# upstream actually tunes for an SM80-MMA kernel.
_BWD_TILE_M = 64
_BWD_TILE_N = 64
_BWD_NUM_THREADS = 128
_BWD_ATOM_LAYOUT = 4


def _round_up(x: int, to: int) -> int:
    return -(-x // to) * to


@functools.cache
def _bwd_kernel_cls(arch: int):
    if arch // 10 == 12:
        from fa4_jax.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120

        return FlashAttentionBackwardSm120
    if arch // 10 == 8:
        from fa4_jax.cute.flash_bwd import FlashAttentionBackwardSm80

        return FlashAttentionBackwardSm80
    raise NotImplementedError(f"fa4 backward wraps the SM80/SM120 kernels only; device is sm_{arch}")


@functools.cache
def _get_bwd(dtype, head_dim, head_dim_v, causal, local, arch):
    stages = 2 if head_dim <= 64 else 1
    return _bwd_kernel_cls(arch)(
        dtype,
        head_dim,
        head_dim_v,
        1,  # qhead_per_kvhead; GQA backward needs a dK/dV accumulator pass
        _BWD_TILE_M,
        _BWD_TILE_N,
        stages,  # num_stages_Q
        stages,  # num_stages_dO
        _BWD_NUM_THREADS,
        False,  # pack_gqa
        causal,
        local,
        False,  # SdP_swapAB
        False,  # dKV_swapAB
        False,  # dQ_swapAB
        _BWD_ATOM_LAYOUT,  # AtomLayoutMSdP
        _BWD_ATOM_LAYOUT,  # AtomLayoutNdKV
        _BWD_ATOM_LAYOUT,  # AtomLayoutMdQ
        V_in_regs=False,
    )


@functools.cache
def _get_bwd_pre(dtype, head_dim, head_dim_v):
    from fa4_jax.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess

    return FlashAttentionBackwardPreprocess(dtype, head_dim, head_dim_v, _BWD_TILE_M)


@functools.cache
def _get_bwd_post(dtype, head_dim, arch):
    from fa4_jax.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess

    return FlashAttentionBackwardPostprocess(
        dtype, head_dim, arch, _BWD_TILE_M, _BWD_NUM_THREADS, _BWD_ATOM_LAYOUT, False
    )


@cute.jit
def _fa4_bwd_pre(stream, tree, mO, mdO, mLSE, mPdPsum, mLSElog2, mdQaccum):
    tree["pre"](
        mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum,
        None, None, None, None, None,  # cu_seqlens_q, seqused_q, dLSE, row_max, scale_p
        cutlass.Float32(tree["scale"]),
        stream=stream,
    )


@cute.jit
def _fa4_bwd(stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum, _mdQaccum_in, mdQaccum, mdK, mdV):
    # dQaccum is passed twice: once as the input holding the zeros preprocess
    # wrote, once as an output. `input_output_aliases` makes XLA hand both
    # names the same buffer, which is what lets the kernel accumulate dQ in
    # place across n-blocks.
    wl, wr = _windows(tree)
    tree["bwd"](
        mQ, mK, mV, mdO, mLSElog2, mPdPsum, mdQaccum, mdK, mdV,
        cutlass.Float32(tree["scale"]),
        None, None, None, None,  # cu_seqlens_q/k, seqused_q/k
        window_size_left=wl,
        window_size_right=wr,
        stream=stream,
    )


@cute.jit
def _fa4_bwd_post(stream, tree, mdQaccum, mdQ):
    tree["post"](mdQaccum, mdQ, cutlass.Float32(tree["scale"]), None, None, None, stream=stream)


# ---------------------------------------------------------------- jvp tangent


@functools.cache
def _get_jvp(dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, mask_mod, n_aux, arch):
    from fa4_jax.cute.flash_jvp import FlashAttentionJvpSm80

    # Six tensor tiles must fit where the forward keeps three; halve tile_m
    # once the head dims outgrow the budget (see flash_jvp.can_implement).
    tile_m = 128 if max(head_dim, head_dim_v) <= 64 else 64
    return FlashAttentionJvpSm80(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        is_causal=causal,
        is_local=local,
        tile_m=tile_m,
        tile_n=64,
        num_threads=128,
        mask_mod=None if mask_mod is None else _cute_mask_mod(mask_mod),
        has_aux_tensors=n_aux > 0,
    )


@cute.jit
def _fa4_jvp(stream, tree, mTO, mMu):
    wl, wr = _windows(tree)
    tree["jvp"](
        tree["q"], tree["k"], tree["v"],
        tree["dq"], tree["dk"], tree["dv"],
        tree["out"], tree["lse"], mTO, mMu, tree["scale"],
        window_size_left=wl, window_size_right=wr,
        aux_data=_aux_data(tree), stream=stream,
    )


# ---------------------------------------------------------------- backend


class _StaticDict(dict):
    """Kernel objects + compile-time scalars for one attention variant.

    A dict so it flattens straight into the cutejax pytree, but hashed by
    identity: `_get_tree` memoizes it, so one variant is always the same
    object and therefore one kernel-cache key.
    """

    __hash__ = object.__hash__  # type: ignore[assignment]
    __eq__ = object.__eq__  # type: ignore[assignment]


@functools.cache
def _get_static(cfg: AttnConfig, dtype, head_dim, head_dim_v, qhead_per_kvhead, arch, kind):
    """The static part of the call tree for (variant, kind); kind picks which
    kernel objects exist so unused ones are never built or compiled."""
    wl, wr = cfg.window
    st = _StaticDict(scale=cfg.scale, wl=wl, wr=wr, n_aux=cfg.n_aux)
    if kind == "fwd":
        st["fa"] = _get_fwd(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            cfg.causal, cfg.local, cfg.mask_mod, cfg.n_aux, arch,
        )
    elif kind == "bwd":
        st["bwd"] = _get_bwd(dtype, head_dim, head_dim_v, cfg.causal, cfg.local, arch)
        st["pre"] = _get_bwd_pre(dtype, head_dim, head_dim_v)
        st["post"] = _get_bwd_post(dtype, head_dim, arch)
    elif kind == "jvp":
        st["jvp"] = _get_jvp(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            cfg.causal, cfg.local, cfg.mask_mod, cfg.n_aux, arch,
        )
    return st


def _validate(cfg: AttnConfig, q, k, v):
    dtype = _DTYPES.get(jnp.dtype(q.dtype))
    if dtype is None:
        raise ValueError(f"FA4 kernels take fp16/bf16 only, got {q.dtype}")
    head_dim, head_dim_v = q.shape[3], v.shape[3]
    if head_dim % 8 or head_dim_v % 8:
        raise ValueError(f"head dims must be multiples of 8, got {head_dim}, {head_dim_v}")
    return dtype, head_dim, head_dim_v, q.shape[2] // k.shape[2]


class CuteBackend:
    def _tree(self, cfg, kind, q, k, v, aux, **extra):
        dtype, hd, hdv, ratio = _validate(cfg, q, k, v)
        tree = dict(_get_static(cfg, dtype, hd, hdv, ratio, _device_arch(), kind))
        tree.update(q=q, k=k, v=v, **extra)
        tree.update({f"aux{i}": a for i, a in enumerate(aux)})
        specs = {"q": _QKV_SPEC, "k": _QKV_SPEC, "v": _QKV_SPEC}
        return tree, specs

    @staticmethod
    def _in_specs(tree, specs):
        # cutejax matches in_specs to array leaves in jax.tree.flatten order,
        # which for a dict is sorted-key order; unlisted leaves (aux tensors,
        # lse) default to fully dynamic.
        return [specs.get(name) for name in sorted(tree) if _is_dynamic(tree[name])]

    def fwd(self, cfg, q, k, v, aux):
        tree, specs = self._tree(cfg, "fwd", q, k, v, aux)
        return cutejax.call(
            _fa4_fwd,
            jax.ShapeDtypeStruct((*q.shape[:3], v.shape[3]), q.dtype),
            tree,
            in_specs=self._in_specs(tree, specs),
            out_specs=_QKV_SPEC,
            name="fa4_fwd",
        )

    def fwd_lse(self, cfg, q, k, v, aux):
        b, s_q, h, _ = q.shape
        tree, specs = self._tree(cfg, "fwd", q, k, v, aux)
        return cutejax.call(
            _fa4_fwd_lse,
            [
                jax.ShapeDtypeStruct((*q.shape[:3], v.shape[3]), q.dtype),
                jax.ShapeDtypeStruct((b, h, s_q), jnp.float32),
            ],
            tree,
            in_specs=self._in_specs(tree, specs),
            out_specs=[_QKV_SPEC, _LSE_SPEC],
            name="fa4_fwd_lse",
        )

    def tangent(self, cfg, q, k, v, out, lse, dq, dk, dv, aux):
        tree, specs = self._tree(
            cfg, "jvp", q, k, v, aux, dq=dq, dk=dk, dv=dv, out=out, lse=lse
        )
        specs.update(dq=_QKV_SPEC, dk=_QKV_SPEC, dv=_QKV_SPEC, out=_QKV_SPEC, lse=_LSE_SPEC)
        b, s_q, h, _ = q.shape
        return cutejax.call(
            _fa4_jvp,
            [
                jax.ShapeDtypeStruct((*q.shape[:3], v.shape[3]), q.dtype),
                jax.ShapeDtypeStruct((b, h, s_q), jnp.float32),
            ],
            tree,
            in_specs=self._in_specs(tree, specs),
            out_specs=[_QKV_SPEC, _LSE_SPEC],
            name="fa4_jvp",
        )

    def vjp(self, cfg, q, k, v, out, lse, d_out, d_lse, aux):
        if cfg.mask_mod is not None:
            raise NotImplementedError(
                "reverse mode with a mask_mod: upstream's SM80/SM120 backward kernel "
                "has no mask_mod support. Use backend='reference', or forward-mode "
                "differentiation (jax.jvp / jax.jacfwd), which supports mask mods."
            )
        if q.shape[2] != k.shape[2]:
            raise NotImplementedError(
                "fa4 backward does not support GQA (it would need a dK/dV accumulator "
                "pass this wrapper omits). Repeat the kv heads out to q heads with "
                "jnp.repeat if you need gradients."
            )
        if d_lse is not None:
            raise NotImplementedError(
                "reverse mode through the lse output is not wired to the cute "
                "backward yet; use backend='reference' if you need d(lse)."
            )
        d_out = d_out.astype(q.dtype)
        b, s_q, h, d = q.shape
        s_q_pad = _round_up(s_q, _BWD_TILE_M)
        d_pad = _round_up(d, 32)
        f32 = jnp.float32
        dtype, hd, hdv, _ = _validate(cfg, q, k, v)
        arch = _device_arch()
        st = _get_static(cfg, dtype, hd, hdv, 1, arch, "bwd")

        dpsum, lse_log2, dq_accum = cutejax.call(
            _fa4_bwd_pre,
            [
                jax.ShapeDtypeStruct((b, h, s_q_pad), f32),
                jax.ShapeDtypeStruct((b, h, s_q_pad), f32),
                jax.ShapeDtypeStruct((b, h, s_q_pad * d_pad), f32),
            ],
            st, out, d_out, lse,
            in_specs=[_QKV_SPEC, _QKV_SPEC, _LSE_SPEC],
            out_specs=_F32_SPEC,
            name="fa4_bwd_pre",
        )

        dq_accum, dk, dv = cutejax.call(
            _fa4_bwd,
            [
                jax.ShapeDtypeStruct(dq_accum.shape, f32),
                jax.ShapeDtypeStruct(k.shape, k.dtype),
                jax.ShapeDtypeStruct(v.shape, v.dtype),
            ],
            st, q, k, v, d_out, lse_log2, dpsum, dq_accum,
            in_specs=[_QKV_SPEC] * 4 + [_F32_SPEC] * 3,
            out_specs=[_F32_SPEC, _QKV_SPEC, _QKV_SPEC],
            input_output_aliases={6: 0},
            name="fa4_bwd",
        )

        dq = cutejax.call(
            _fa4_bwd_post,
            jax.ShapeDtypeStruct((b, s_q, h, d), q.dtype),
            st, dq_accum,
            in_specs=[_F32_SPEC],
            out_specs=_QKV_SPEC,
            name="fa4_bwd_post",
        )
        return dq, dk, dv


def _is_dynamic(leaf) -> bool:
    """Whether a tree leaf is a jax array (dynamic data) rather than static."""
    return isinstance(leaf, (jax.Array, jax.core.Tracer))


register_backend("cute", CuteBackend())
