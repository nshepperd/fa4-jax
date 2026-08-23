"""The cute backend: vendored FA4 kernels driven through cutejax.

A port of minimax-music3's fa4.py restructured behind the Backend protocol,
plus flex-attention mask mods: the MaskMod (a callable pytree) arrives as its
treedef plus a tuple of array leaves; the treedef is wrapped into a
``@cute.jit`` callable that rebuilds the mask with the kernel's CuTe aux
tensors as leaves and calls it (scalar Int32 indices in, Boolean SSA out), so
the predicate becomes codegen keyed by the mask's class + static fields,
while its array fields stay runtime data.

Kernel-side restrictions surface here as NotImplementedError:

- backward (reverse mode): no lse cotangent; mask_mod works (a local addition
  to the vendored backward). GQA works: the kernel atomically accumulates dK/dV
  in fp32 across the q heads sharing a kv head, and two extra postprocess
  launches convert them (see the backward section). The JVP tangent kernel
  supports all of it.
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
def _cute_mask_mod(mask_def: jax.tree_util.PyTreeDef):
    """Stage a MaskMod treedef as the cute callable the kernels expect.

    The kernel's aux tensors are the mask's array leaves in flatten order, so
    unflattening them through the treedef yields the mask with CuTe tensors
    in its array fields. The kernels hand the predicate size-1 SSA vectors;
    the MaskMod contract is scalar indices and a scalar Boolean, so scalarize
    on the way in and re-wrap on the way out. Cached per treedef (class +
    static fields): the returned function object is baked into kernel configs
    and must be stable.
    """
    from fa4_jax.cute.utils import scalar_to_ssa

    @cute.jit
    def mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        # aux_tensors is None (not ()) when the kernel has has_aux_tensors=False.
        leaves = () if aux_tensors is None else tuple(aux_tensors)
        mask: MaskMod = mask_def.unflatten(leaves)
        pred = mask(batch_idx[0], head_idx[0], q_idx[0], kv_idx[0])
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
        f"fa4-jax wraps the SM80/SM100/SM120 kernels only; device is sm_{arch}. "
        "Hopper (sm_90) has its own FA4 kernels that are not vendored here."
    )


def _fwd_tile_mn(arch: int, head_dim: int) -> tuple[int, int]:
    """Tile sizes matching flash_attn.cute.interface's per-arch heuristic."""
    if arch // 10 == 12 and head_dim <= 64:
        return 128, 128  # 48 KB of the 99 KB SMEM budget
    return 128, 64


@functools.cache
def _get_fwd(dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, mask_mod, n_aux, arch):
    if arch // 10 == 10:
        # Datacenter Blackwell (tcgen05). Fixed knobs where upstream's
        # interface.py has shape-dependent heuristics, since batch/seqlen stay
        # symbolic here: q_stage=2 keeps two q-tiles in flight (upstream drops
        # to 1 only when seqlen_q fits one tile -- with a short seqlen the
        # second stage is fully masked, which is correct, just wasted work);
        # 2CTA and the CLC scheduler are off (both need cluster launches).
        # Dense tiles self-balance under the static persistent scheduler;
        # causal/local use the LPT single-tile scheduler, upstream's choice
        # when CLC is off.
        from fa4_jax.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

        return FlashAttentionForwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            pack_gqa=False,
            m_block_size=128,
            n_block_size=128,
            q_stage=2,
            is_static_persistent=not (causal or local),
            mask_mod=None if mask_mod is None else _cute_mask_mod(mask_mod),
            has_aux_tensors=n_aux > 0,
            use_2cta_instrs=False,
            use_clc_scheduler=False,
        )
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
# With GQA (qhead_per_kvhead > 1) the kernel cannot write dK/dV directly:
# every q head sharing a kv head contributes, so it atomically accumulates
# into fp32 dKaccum/dVaccum laid out like dQaccum (zeroed here with
# jnp.zeros; the preprocess only zeroes dQaccum) and leaves the softmax scale
# off dK. The same postprocess kernel then converts both: dK with the scale,
# dV with scale 1. Its tile is _bwd_tile_m, which matches the kv tile because
# the backward's q and kv tiles are equal on every path here.
#
# Tile config: upstream has no sm_8x branch for the backward -- arch 8 falls
# through to the SM100 settings, which ask for 128x128 tiles that will not fit
# sm_86's 100 KB of SMEM. We use the SM120 config (64x64, 128 threads) on both
# sm_8x and sm_120 instead: conservative, fits everywhere, and it is the only
# one upstream actually tunes for an SM80-MMA kernel. SM100 (tcgen05) uses
# upstream's 128x128.
_BWD_NUM_THREADS = 128
_BWD_ATOM_LAYOUT = 4


def _bwd_use_sm100(arch: int, head_dim: int, head_dim_v: int) -> bool:
    """Whether the tcgen05 backward serves this head config.

    The 1-CTA SM100 kernel takes head dims up to 128, but 4.0.0b26's
    mixed-dims path (head_dim != head_dim_v) miscompiles (a tmem layout
    division error), so those fall back to the SM80-MMA kernel, which
    compiles fine for sm_100 -- cp.async and mma.sync exist there too.
    """
    return arch // 10 == 10 and head_dim == head_dim_v and head_dim <= 128


def _bwd_tile_m(arch: int, head_dim: int, head_dim_v: int) -> int:
    """The backward's (square) tile size; also the accumulator padding unit."""
    return 128 if _bwd_use_sm100(arch, head_dim, head_dim_v) else 64


def _round_up(x: int, to: int) -> int:
    return -(-x // to) * to


@functools.cache
def _bwd_kernel_cls(arch: int):
    if arch // 10 == 12:
        from fa4_jax.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120

        return FlashAttentionBackwardSm120
    if arch // 10 in (8, 10):  # sm_100 uses this as the mixed-head-dims fallback
        from fa4_jax.cute.flash_bwd import FlashAttentionBackwardSm80

        return FlashAttentionBackwardSm80
    raise NotImplementedError(
        f"fa4 backward wraps the SM80/SM100/SM120 kernels only; device is sm_{arch}"
    )


@functools.cache
def _get_bwd(dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, mask_mod, n_aux, arch):
    tile = _bwd_tile_m(arch, head_dim, head_dim_v)
    if _bwd_use_sm100(arch, head_dim, head_dim_v):
        from fa4_jax.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100

        return FlashAttentionBackwardSm100(
            head_dim,
            head_dim_v,
            is_causal=causal,
            is_local=local,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=tile,
            tile_n=tile,
            deterministic=False,
            spt=False,
            cluster_size=1,
            use_2cta_instrs=False,
            mask_mod=None if mask_mod is None else _cute_mask_mod(mask_mod),
            has_aux_tensors=n_aux > 0,
        )
    stages = 2 if head_dim <= 64 else 1
    return _bwd_kernel_cls(arch)(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        tile,  # tile_m
        tile,  # tile_n
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
        mask_mod=None if mask_mod is None else _cute_mask_mod(mask_mod),
        has_aux_tensors=n_aux > 0,
    )


@functools.cache
def _get_bwd_pre(dtype, head_dim, head_dim_v, arch):
    from fa4_jax.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess

    return FlashAttentionBackwardPreprocess(
        dtype, head_dim, head_dim_v, _bwd_tile_m(arch, head_dim, head_dim_v)
    )


@functools.cache
def _get_bwd_post(dtype, head_dim, arch, sm100: bool):
    """Postprocess for one accumulator; head_dim is the converted tensor's
    (dQ/dK take the problem's head_dim, dV takes head_dim_v).

    The arch/block/atom parameters select the accumulator element layout,
    which must match the kernel that wrote it: the tcgen05 backward writes
    the SM100 layout (AtomLayoutMdQ=1, upstream's default), the SM80-MMA
    kernel the sm_8x/12x one -- including when it runs on sm_100 as the
    mixed-head-dims fallback, hence the layout arch, not the device arch.
    """
    from fa4_jax.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess

    if sm100:
        layout_arch, atom_layout, block = arch, 1, 128
    else:
        layout_arch, atom_layout, block = (120 if arch // 10 == 10 else arch), _BWD_ATOM_LAYOUT, 64
    return FlashAttentionBackwardPostprocess(
        dtype, head_dim, layout_arch, block, _BWD_NUM_THREADS, atom_layout, False
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
def _fa4_bwd(stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum, _mdQaccum_in, aux, mdQaccum, mdK, mdV):
    # dQaccum is passed twice: once as the input holding the zeros preprocess
    # wrote, once as an output. `input_output_aliases` makes XLA hand both
    # names the same buffer, which is what lets the kernel accumulate dQ in
    # place across n-blocks. `aux` is the (possibly empty) tuple of mask_mod
    # aux tensors.
    from fa4_jax.cute.utils import AuxData

    wl, wr = _windows(tree)
    tree["bwd"](
        mQ, mK, mV, mdO, mLSElog2, mPdPsum, mdQaccum, mdK, mdV,
        cutlass.Float32(tree["scale"]),
        None, None, None, None,  # cu_seqlens_q/k, seqused_q/k
        window_size_left=wl,
        window_size_right=wr,
        aux_data=AuxData(tensors=aux) if cutlass.const_expr(len(aux) > 0) else AuxData(),
        stream=stream,
    )


@cute.jit
def _fa4_bwd_gqa(
    stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum, _mdQaccum_in, _mdKaccum_in, _mdVaccum_in, aux,
    mdQaccum, mdKaccum, mdVaccum,
):
    """GQA variant: the kernel's dK/dV slots take fp32 accumulators (aliased
    in place like dQaccum) instead of the final dK/dV."""
    from fa4_jax.cute.utils import AuxData

    wl, wr = _windows(tree)
    tree["bwd"](
        mQ, mK, mV, mdO, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum,
        cutlass.Float32(tree["scale"]),
        None, None, None, None,  # cu_seqlens_q/k, seqused_q/k
        window_size_left=wl,
        window_size_right=wr,
        aux_data=AuxData(tensors=aux) if cutlass.const_expr(len(aux) > 0) else AuxData(),
        stream=stream,
    )


@cute.jit
def _fa4_bwd_post(stream, tree, mdQaccum, mdQ):
    """dQaccum -> dQ (or, under GQA, dKaccum -> dK: same head_dim, same scale)."""
    tree["post"](mdQaccum, mdQ, cutlass.Float32(tree["scale"]), None, None, None, stream=stream)


@cute.jit
def _fa4_bwd_post_dv(stream, tree, mdVaccum, mdV):
    """dVaccum -> dV under GQA: head_dim_v tiles, and no softmax scale."""
    tree["post_v"](mdVaccum, mdV, cutlass.Float32(1.0), None, None, None, stream=stream)


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
    kernel objects exist so unused ones are never built or compiled.

    Native causal/window options combine with a mask_mod directly: the
    kernels keep the band's block-range skipping and AttentionMask runs the
    mod as a second masking pass over the visited tiles (a local change to
    the vendored mask.py -- upstream's branch structure would silently drop
    the mod in that combination).
    """
    causal, local, (wl, wr), mask_mod = cfg.causal, cfg.local, cfg.window, cfg.mask
    st = _StaticDict(scale=cfg.scale, wl=wl, wr=wr, n_aux=cfg.n_aux)
    if kind == "fwd":
        st["fa"] = _get_fwd(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            causal, local, mask_mod, cfg.n_aux, arch,
        )
    elif kind == "bwd":
        st["bwd"] = _get_bwd(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            causal, local, mask_mod, cfg.n_aux, arch,
        )
        sm100 = _bwd_use_sm100(arch, head_dim, head_dim_v)
        st["pre"] = _get_bwd_pre(dtype, head_dim, head_dim_v, arch)
        st["post"] = _get_bwd_post(dtype, head_dim, arch, sm100)
        if qhead_per_kvhead > 1:
            st["post_v"] = _get_bwd_post(dtype, head_dim_v, arch, sm100)
    elif kind == "jvp":
        st["jvp"] = _get_jvp(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            causal, local, mask_mod, cfg.n_aux, arch,
        )
    return st


@functools.cache
def _can_implement(kind: str, dtype, head_dim: int, head_dim_v: int, causal: bool, arch: int) -> bool:
    """Ask the kernel class whether our tile config fits (SMEM budget, head
    dim limits); a launch that overflows SMEM would otherwise fail *inside*
    CUDA and leave the context in an error state."""
    if kind == "fwd":
        if arch // 10 == 10:
            # The tcgen05 forward has no can_implement; keep the family's
            # public envelope (head dims <= 128 -- upstream's "standard
            # range" for SM100, larger needs 2-CTA MMAs we don't launch).
            return head_dim <= 128 and head_dim_v <= 128
        tile_m, tile_n = _fwd_tile_mn(arch, head_dim)
        return _fwd_kernel_cls(arch).can_implement(
            dtype, head_dim, head_dim_v, tile_m, tile_n, 1, 128, causal
        )
    if kind == "bwd":
        if _bwd_use_sm100(arch, head_dim, head_dim_v):
            # No can_implement on the tcgen05 backward; _bwd_use_sm100 already
            # bounds the head dims, its constructor asserts the rest.
            return True
        # sm_100 mixed-dims fallback runs the SM80-MMA kernel: use its check.
        tile = _bwd_tile_m(arch, head_dim, head_dim_v)
        stages = 2 if head_dim <= 64 else 1
        return _bwd_kernel_cls(arch).can_implement(
            dtype, head_dim, head_dim_v, tile, tile, stages, stages,
            _BWD_NUM_THREADS, causal,
        )
    if kind == "jvp":
        from fa4_jax.cute.flash_jvp import FlashAttentionJvpSm80

        tile_m = 128 if max(head_dim, head_dim_v) <= 64 else 64
        return FlashAttentionJvpSm80.can_implement(dtype, head_dim, head_dim_v, tile_m, 64, 128)
    raise ValueError(kind)


def _validate(cfg: AttnConfig, kind: str, q, k, v):
    dtype = _DTYPES.get(jnp.dtype(q.dtype))
    if dtype is None:
        raise ValueError(f"FA4 kernels take fp16/bf16 only, got {q.dtype}")
    head_dim, head_dim_v = q.shape[3], v.shape[3]
    if head_dim % 8 or head_dim_v % 8:
        raise ValueError(f"head dims must be multiples of 8, got {head_dim}, {head_dim_v}")
    arch = _device_arch()
    if not _can_implement(kind, dtype, head_dim, head_dim_v, cfg.causal, arch):
        raise ValueError(
            f"FA4 {kind} kernel cannot run head_dim={head_dim}, head_dim_v={head_dim_v} "
            f"({q.dtype}) on sm_{arch}: tile does not fit the shared-memory budget "
            "or the head dims exceed what the kernel supports (<= 128 on sm_8x/sm_120)"
        )
    return dtype, head_dim, head_dim_v, q.shape[2] // k.shape[2], arch


class CuteBackend:
    def _tree(self, cfg, kind, q, k, v, aux, **extra):
        dtype, hd, hdv, ratio, arch = _validate(cfg, kind, q, k, v)
        tree = dict(_get_static(cfg, dtype, hd, hdv, ratio, arch, kind))
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
        if d_lse is not None:
            raise NotImplementedError(
                "reverse mode through the lse output is not wired to the cute "
                "backward yet; use backend='reference' if you need d(lse)."
            )
        d_out = d_out.astype(q.dtype)
        b, s_q, h, d = q.shape
        _, s_kv, h_kv, d_v = v.shape
        d_pad = _round_up(d, 32)
        f32 = jnp.float32
        dtype, hd, hdv, ratio, arch = _validate(cfg, "bwd", q, k, v)
        bwd_tile = _bwd_tile_m(arch, hd, hdv)
        s_q_pad = _round_up(s_q, bwd_tile)
        st = _get_static(cfg, dtype, hd, hdv, ratio, arch, "bwd")

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

        if ratio == 1:
            dq_accum, dk, dv = cutejax.call(
                _fa4_bwd,
                [
                    jax.ShapeDtypeStruct(dq_accum.shape, f32),
                    jax.ShapeDtypeStruct(k.shape, k.dtype),
                    jax.ShapeDtypeStruct(v.shape, v.dtype),
                ],
                st, q, k, v, d_out, lse_log2, dpsum, dq_accum, tuple(aux),
                in_specs=[_QKV_SPEC] * 4 + [_F32_SPEC] * 3 + [None] * len(aux),
                out_specs=[_F32_SPEC, _QKV_SPEC, _QKV_SPEC],
                input_output_aliases={6: 0},
                name="fa4_bwd",
            )
        else:
            s_kv_pad = _round_up(s_kv, bwd_tile)
            dk_accum = jnp.zeros((b, h_kv, s_kv_pad * d_pad), f32)
            dv_accum = jnp.zeros((b, h_kv, s_kv_pad * _round_up(d_v, 32)), f32)
            dq_accum, dk_accum, dv_accum = cutejax.call(
                _fa4_bwd_gqa,
                [
                    jax.ShapeDtypeStruct(dq_accum.shape, f32),
                    jax.ShapeDtypeStruct(dk_accum.shape, f32),
                    jax.ShapeDtypeStruct(dv_accum.shape, f32),
                ],
                st, q, k, v, d_out, lse_log2, dpsum, dq_accum, dk_accum, dv_accum, tuple(aux),
                in_specs=[_QKV_SPEC] * 4 + [_F32_SPEC] * 5 + [None] * len(aux),
                out_specs=[_F32_SPEC] * 3,
                input_output_aliases={6: 0, 7: 1, 8: 2},
                name="fa4_bwd_gqa",
            )
            dk = cutejax.call(
                _fa4_bwd_post,
                jax.ShapeDtypeStruct(k.shape, k.dtype),
                st, dk_accum,
                in_specs=[_F32_SPEC],
                out_specs=_QKV_SPEC,
                name="fa4_bwd_post_dk",
            )
            dv = cutejax.call(
                _fa4_bwd_post_dv,
                jax.ShapeDtypeStruct(v.shape, v.dtype),
                st, dv_accum,
                in_specs=[_F32_SPEC],
                out_specs=_QKV_SPEC,
                name="fa4_bwd_post_dv",
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
