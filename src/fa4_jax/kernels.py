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
- varlen (packed) attention: forward, backward, and the JVP tangent kernel
  all run the native cu_seqlens/seqused path.
- fp16/bf16 only, head_dim % 8 == 0 (validated with clear errors).

Layout note (from fa4.py): heads and head_dim are baked into the kernel (they
pick tile sizes and layouts); batch and seqlen stay symbolic, so one compile
serves every length.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import cutejax
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute

from fa4_jax.attention import AttnConfig, register_backend
from fa4_jax.masks import MaskMod

_QKV_SPEC = cutejax.ArraySpec(static_dims=(2, 3))
# Packed varlen layout (total_tokens, heads, head_dim): heads/head_dim static
# (they pick tile sizes and layouts), the token axis symbolic. The cu_seqlens
# arrays stay fully dynamic, so one compile serves every batch size too.
_QKV3_SPEC = cutejax.ArraySpec(static_dims=(1, 2))

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
    """The SM80-MMA (cp.async + mma.sync) forward family. SM120 differs from
    SM80 only in the shared-memory capacity check, but that check is what
    keeps 128x128 tiles from silently overflowing 99 KB. The tcgen05 (sm_100)
    and wgmma (sm_90) forwards are separate classes, picked in _get_fwd."""
    if arch // 10 == 12:
        from fa4_jax.cute.flash_fwd_sm120 import FlashAttentionForwardSm120

        return FlashAttentionForwardSm120
    if arch // 10 == 8:
        from fa4_jax.cute.flash_fwd import FlashAttentionForwardSm80

        return FlashAttentionForwardSm80
    raise NotImplementedError(
        f"fa4-jax wraps the SM80/SM90/SM100/SM120 kernel families only; got sm_{arch}"
    )


def _fwd_tile_mn(arch: int, dtype, head_dim: int, head_dim_v: int) -> tuple[int, int]:
    """Tile sizes matching flash_attn.cute.interface's per-arch heuristic,
    plus a 64x64 tile for large head dims on 99 KB parts: 128x64 needs 128 KB
    of SMEM at hd 256 (and its O accumulator would be 256 fp32 registers per
    thread), while 64x64 is 96 KB -- FA2's hd256 config on sm86/89.

    sm_12x switches on the larger head dim (O, head_dim_v wide, drives
    register pressure). sm_8x keeps 128x64 wherever it fits -- sm_80's
    163 KB takes it up to hd 256 -- and drops to 64x64 where it doesn't
    (sm_86/89)."""
    if arch // 10 == 12:
        hd = max(head_dim, head_dim_v)
        if hd <= 64:
            return 128, 128  # 48 KB of the 99 KB SMEM budget
        if hd <= 128:
            return 128, 64
        return 64, 64
    if arch // 10 == 8:
        import cutlass.utils as utils_basic

        smem = utils_basic.get_smem_capacity_in_bytes(f"sm_{arch}")
        if not _fwd_kernel_cls(arch).can_implement(
            dtype, head_dim, head_dim_v, 128, 64, 1, 128, False, smem_capacity=smem
        ):
            return 64, 64
    return 128, 64


# ---------------------------------------------------------------- hopper configs
#
# The sm_90 (wgmma + TMA) kernels are tuned per head dim: upstream's
# interface.py carries the tables (FA3's C++ configs, re-benchmarked on H100
# SXM for the Python kernels), reproduced here minus the block-sparse cases.
# Unlike the other families these tiles are not square, so the backward's
# accumulator padding, preprocess and postprocess must follow (tile_m, tile_n)
# rather than one tile size -- see _BwdPlan.


@dataclass(frozen=True)
class _FwdCfgSm90:
    tile_m: int
    tile_n: int
    mma_pv_is_rs: bool  # PV MMA reads P from registers (False: staged through smem)
    intra_wg_overlap: bool


def _fwd_cfg_sm90(head_dim: int, head_dim_v: int, causal: bool, local: bool) -> _FwdCfgSm90:
    """interface._tile_size_fwd_sm90, keyed on the *larger* head dim: upstream
    keys on head_dim and sizes the O accumulator for head_dim_v ~ head_dim,
    but a much larger head_dim_v under the 192-row / 3-warpgroup tiles would
    blow the 160-register budget of that layout."""
    hd = max(head_dim, head_dim_v)
    if hd <= 64:
        return _FwdCfgSm90(192, 128, True, True)
    if hd <= 96:
        # Upstream: RS is catastrophic with 192-row tiles at this head dim.
        return _FwdCfgSm90(192, 128 if (causal or local) else 144, False, True)
    if hd <= 128:
        return _FwdCfgSm90(128, 128, True, True)
    if hd <= 192:
        return _FwdCfgSm90(128, 96 if local else (128 if head_dim_v <= 128 else 112), True, True)
    return _FwdCfgSm90(128, 64 if local else 80, True, True)


@dataclass(frozen=True)
class _BwdCfgSm90:
    """One row of interface._tile_size_bwd_sm90 (FA3's C++ configs)."""

    tile_m: int
    tile_n: int
    Q_stage: int
    dO_stage: int
    PdS_stage: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warpgroups; the kernel adds one producer warpgroup
    dQ_single_wg: bool = False  # only warpgroup 0 computes dQ

    @property
    def num_threads(self) -> int:
        return (self.num_wg + 1) * 128


def _bwd_cfg_sm90(head_dim: int, head_dim_v: int, causal: bool, local: bool) -> _BwdCfgSm90:
    if head_dim <= 64:
        return _BwdCfgSm90(128, 128, 2, 2, 2, True, False, False, 1, 2, 2)
    if head_dim <= 96:
        return _BwdCfgSm90(64, 128, 2, 2, 2, True, False, False, 1, 2, 1, dQ_single_wg=True)
    if head_dim <= 128:
        tile_m = 64 if (causal or local) else 80
        return _BwdCfgSm90(tile_m, 128, 2, 2, 2, True, False, tile_m % 64 != 0, 1, 2, 1)
    if head_dim <= 192:
        return _BwdCfgSm90(64, 96, 2, 2 if head_dim_v <= 128 else 1, 1, False, True, False, 1, 2, 1)
    return _BwdCfgSm90(64, 64, 1, 1, 1, False, False, False, 1, 1, 1)


def _sm90_bwd_kwargs(c: _BwdCfgSm90, causal: bool, local: bool) -> dict:
    """Constructor kwargs of FlashAttentionBackwardSm90 for one config row
    (shared by can_implement and the real construction)."""
    return dict(
        is_causal=causal,
        is_local=local,
        deterministic=False,
        tile_m=c.tile_m,
        tile_n=c.tile_n,
        Q_stage=c.Q_stage,
        dO_stage=c.dO_stage,
        PdS_stage=c.PdS_stage,
        SdP_swapAB=c.SdP_swapAB,
        dKV_swapAB=c.dKV_swapAB,
        dQ_swapAB=c.dQ_swapAB,
        AtomLayoutMSdP=c.AtomLayoutMSdP,
        AtomLayoutNdKV=c.AtomLayoutNdKV,
        AtomLayoutMdQ=c.AtomLayoutMdQ,
        num_threads=c.num_threads,
        V_in_regs=False,
        dQ_single_wg=c.dQ_single_wg,
    )


@functools.cache
def _get_fwd(dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, mask_mod, n_aux, arch, varlen=False):
    if arch // 10 == 9:
        # Hopper (wgmma + TMA). num_stages=2 is upstream's interface setting;
        # the kernel derives its thread count from the tile (tile_m // 64 MMA
        # warpgroups plus one producer warpgroup), so the value passed is
        # nominal. Varlen needs no constructor flag: the kernel switches to
        # the varlen scheduler and a ragged TMA O store on its own when it
        # sees cu_seqlens.
        from fa4_jax.cute.flash_fwd_sm90 import FlashAttentionForwardSm90

        c = _fwd_cfg_sm90(head_dim, head_dim_v, causal, local)
        return FlashAttentionForwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            pack_gqa=False,
            tile_m=c.tile_m,
            tile_n=c.tile_n,
            num_stages=2,
            num_threads=(c.tile_m // 64 + 1) * 128,
            Q_in_regs=False,
            intra_wg_overlap=c.intra_wg_overlap,
            mma_pv_is_rs=c.mma_pv_is_rs,
            mask_mod=None if mask_mod is None else _cute_mask_mod(mask_mod),
            has_aux_tensors=n_aux > 0,
        )
    if arch // 10 == 10:
        # Datacenter Blackwell (tcgen05). Fixed knobs where upstream's
        # interface.py has shape-dependent heuristics, since batch/seqlen stay
        # symbolic here: q_stage=2 keeps two q-tiles in flight (upstream drops
        # to 1 only when seqlen_q fits one tile -- with a short seqlen the
        # second stage is fully masked, which is correct, just wasted work);
        # 2CTA and the CLC scheduler are off (both need cluster launches).
        # Dense tiles self-balance under the static persistent scheduler;
        # causal/local use the LPT single-tile scheduler, upstream's choice
        # when CLC is off. Varlen must be declared at construction time:
        # is_varlen_q switches the O epilogue off TMA (a TMA store cannot
        # follow the ragged per-sequence offsets -- with the flag unset the
        # kernel scribbles O nondeterministically while the LSE, written by
        # the non-TMA path, stays correct), and is_static_persistent off so
        # the kernel takes SingleTileVarlenScheduler (per-batch m-block
        # counts decoded from cu_seqlens) instead of visiting every batch's
        # worst-case tile range.
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
            is_static_persistent=not (causal or local or varlen),
            is_varlen_q=varlen,
            mask_mod=None if mask_mod is None else _cute_mask_mod(mask_mod),
            has_aux_tensors=n_aux > 0,
            use_2cta_instrs=False,
            use_clc_scheduler=False,
        )
    tile_m, tile_n = _fwd_tile_mn(arch, dtype, head_dim, head_dim_v)
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
    # The seqlen keys exist only in varlen trees; dict membership is a
    # trace-time constant, so the non-varlen compile is unchanged.
    wl, wr = _windows(tree)
    tree["fa"](
        tree["q"], tree["k"], tree["v"], mO, None, tree["scale"],
        mCuSeqlensQ=tree.get("cu_q"), mCuSeqlensK=tree.get("cu_k"),
        mSeqUsedQ=tree.get("su_q"), mSeqUsedK=tree.get("su_k"),
        window_size_left=wl, window_size_right=wr,
        aux_data=_aux_data(tree), stream=stream,
    )


@cute.jit
def _fa4_fwd_lse(stream, tree, mO, mLSE):
    """Same kernel, but also writing the log-sum-exp the tangent/backward need."""
    wl, wr = _windows(tree)
    tree["fa"](
        tree["q"], tree["k"], tree["v"], mO, mLSE, tree["scale"],
        mCuSeqlensQ=tree.get("cu_q"), mCuSeqlensK=tree.get("cu_k"),
        mSeqUsedQ=tree.get("su_q"), mSeqUsedK=tree.get("su_k"),
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
# dV with scale 1. Those run with the kv tile (tile_n) and the dK/dV MMA
# layout, which differ from dQ's on sm_90 -- hence separate "post_k"/"post_v"
# kernel objects next to "post" (see _bwd_post_params).
#
# Tile config: upstream has no sm_8x branch for the backward -- arch 8 falls
# through to the SM100 settings, which ask for 128x128 tiles that will not fit
# sm_86's 100 KB of SMEM. We use the SM120 config (64x64, 128 threads) on both
# sm_8x and sm_120 instead: conservative, and it is the only one upstream
# actually tunes for an SM80-MMA kernel. It stops at head dim ~160 on 99 KB
# parts, so larger heads step down a ladder of smaller-tile_n rows
# (_bwd_cfgs_sm80). SM100 (tcgen05) uses upstream's 128x128; SM90 (wgmma)
# its per-head-dim table (_bwd_cfg_sm90).
_BWD_NUM_THREADS = 128  # the tcgen05 postprocess's thread count


@dataclass(frozen=True)
class _BwdCfgSm80:
    """One SM80-MMA backward config. The atom layouts give the warps along M
    for the S/dP and dQ MMAs and along N (the kv tile) for dK/dV."""

    tile_m: int
    tile_n: int
    num_threads: int
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    V_in_regs: bool
    num_stages: int  # for both Q and dO


def _bwd_cfg_sm80_tiles(c: _BwdCfgSm80, head_dim: int, head_dim_v: int) -> bool:
    """Whether the row's three tiled MMAs (warps laid out per its atom
    layouts, 16x16 per warp, as FlashAttentionBackwardSm80 builds them) tile
    the problem: S/dP over (tile_m, tile_n), dK/dV over (tile_n, head dims),
    dQ over (tile_m, head_dim). Head dims are the kernel's 32-padded ones.
    A row that doesn't tile fails at compile time, not in can_implement."""
    warps = c.num_threads // 32
    hd, hdv = _round_up(head_dim, 32), _round_up(head_dim_v, 32)
    sdp_m, dkv_m, dq_m = (16 * a for a in (c.AtomLayoutMSdP, c.AtomLayoutNdKV, c.AtomLayoutMdQ))
    sdp_n, dkv_n, dq_n = (16 * (warps // a) for a in (c.AtomLayoutMSdP, c.AtomLayoutNdKV, c.AtomLayoutMdQ))
    return (
        c.tile_m % sdp_m == 0 and c.tile_n % sdp_n == 0
        and c.tile_n % dkv_m == 0 and hd % dkv_n == 0 and hdv % dkv_n == 0
        and c.tile_m % dq_m == 0 and hd % dq_n == 0
    )


def _bwd_cfgs_sm80(head_dim: int, head_dim_v: int) -> tuple[_BwdCfgSm80, ...]:
    """Candidate SM80-MMA backward configs, best first; _bwd_plan takes the
    first that fits the device's SMEM.

    64x64 / 4 warps is the long-standing default (fits hd <= 160 in 99 KB).
    Past that, FA2's sm86/89 hd256 config: 64x32, 8 warps, V held in
    registers so sV can alias sQ (1.91 ms vs 2.47 ms for 64x16 on the
    Qwen3.5-9B hd256 fwd+bwd, sm_120, s=2048). Its dK/dV MMA spans 8x16 = 128
    head-dim columns, so it needs head dims that are multiples of 128; hd 192
    takes 64x32 / 8 warps with V in SMEM and a 2x4 dK/dV warp grid (1.33 ms
    vs 1.66 ms for 64x32 / 4 warps, hd192 fwd+bwd on sm_120 -- and 64x32
    with 4 warps fails libNVVM on sm_86 for every hd above 128, and spills
    with V in registers). 64x16 / 4 warps, then 32x32,
    are the last resorts; rows that don't tile the head dims are skipped."""
    rows = (
        _BwdCfgSm80(64, 64, 128, 4, 4, 4, False, 2 if head_dim <= 64 else 1),
        _BwdCfgSm80(64, 32, 256, 4, 1, 2, True, 1),
        _BwdCfgSm80(64, 32, 256, 4, 2, 2, False, 1),
        _BwdCfgSm80(64, 16, 128, 4, 1, 4, False, 1),
        _BwdCfgSm80(32, 32, 128, 2, 2, 2, False, 1),
    )
    return tuple(c for c in rows if _bwd_cfg_sm80_tiles(c, head_dim, head_dim_v))


@dataclass(frozen=True)
class _BwdPlan:
    """Which backward kernel serves a head config, plus the tile geometry that
    the fp32 accumulators, the preprocess and the postprocess must agree on
    (dQaccum is padded per tile_m rows, dK/dV accum per tile_n).

    ``family`` is 100 (tcgen05), 90 (wgmma) or 80 (the SM80-MMA kernel:
    native on sm_8x/sm_12x, and the fallback on sm_90/sm_100 for head configs
    their native kernels cannot take -- cp.async and mma.sync exist there
    too). The wgmma and SM80-MMA families carry their full config row in
    ``sm90`` / ``sm80``.
    """

    family: int
    tile_m: int
    tile_n: int
    sm90: _BwdCfgSm90 | None = None
    sm80: _BwdCfgSm80 | None = None


def _round_up(x: int, to: int) -> int:
    return -(-x // to) * to


@functools.cache
def _bwd_kernel_cls(arch: int):
    """The SM80-MMA backward family: the SM120 subclass on sm_12x (its default
    SMEM budget), the SM80 class everywhere else -- including sm_90/sm_100,
    where it is the fallback for what the native kernels reject."""
    if arch // 10 == 12:
        from fa4_jax.cute.flash_bwd_sm120 import FlashAttentionBackwardSm120

        return FlashAttentionBackwardSm120
    if arch // 10 in (8, 9, 10, 11):
        from fa4_jax.cute.flash_bwd import FlashAttentionBackwardSm80

        return FlashAttentionBackwardSm80
    raise NotImplementedError(
        f"fa4 backward wraps the SM80/SM90/SM100/SM120 kernel families only; got sm_{arch}"
    )


@functools.cache
def _bwd_plan(
    dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, arch
) -> _BwdPlan | None:
    """Pick the backward kernel for device ``sm_{arch}``, or
    None if nothing fits (the caller raises a clean ValueError).

    - sm_100: the 1-CTA tcgen05 kernel takes head dims up to 128, but
      4.0.0b26's mixed-dims path (head_dim != head_dim_v) miscompiles (a tmem
      layout division error), and its dQ reduction works in 32-column chunks
      (asserts the 16-padded head dim is a multiple of 32, so hd 16/48 can't
      build); those fall through to the SM80-MMA kernel.
    - sm_90: the wgmma kernel with upstream's config row for the head dim,
      when its SharedStorage fits the device (upstream's largest configs sit
      exactly at the 227 KB limit) and the config is legal (GQA needs equal
      head dims); otherwise the SM80-MMA fallback. The row above hd 192 is
      upstream's hd256 config and only builds for head dims that are
      multiples of 64 (hd 224 fails to partition its SMEM copy).
    - else the SM80-MMA kernel, with the first _bwd_cfgs_sm80 row whose
      SharedStorage fits the device.
    """
    import cutlass.utils as utils_basic

    smem = utils_basic.get_smem_capacity_in_bytes(f"sm_{arch}")
    if (
        arch // 10 == 10
        and head_dim == head_dim_v
        and head_dim <= 128
        and _round_up(head_dim, 16) % 32 == 0
    ):
        # No can_implement on the tcgen05 backward; the head-dim bound above
        # is its envelope and the constructor asserts the rest.
        return _BwdPlan(100, 128, 128)
    big_ok = max(head_dim, head_dim_v) <= 192 or (head_dim % 64 == 0 and head_dim_v % 64 == 0)
    if arch // 10 == 9 and big_ok:
        from fa4_jax.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90

        c = _bwd_cfg_sm90(head_dim, head_dim_v, causal, local)
        if FlashAttentionBackwardSm90.can_implement(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            **_sm90_bwd_kwargs(c, causal, local), smem_capacity=smem,
        ):
            return _BwdPlan(90, c.tile_m, c.tile_n, c)
    for c in _bwd_cfgs_sm80(head_dim, head_dim_v):
        if _bwd_kernel_cls(arch).can_implement(
            dtype, head_dim, head_dim_v, c.tile_m, c.tile_n, c.num_stages, c.num_stages,
            c.num_threads, causal, V_in_regs=c.V_in_regs, smem_capacity=smem,
        ):
            return _BwdPlan(80, c.tile_m, c.tile_n, sm80=c)
    return None


@functools.cache
def _get_bwd(
    dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, mask_mod, n_aux, arch, plan: _BwdPlan
):
    mm = None if mask_mod is None else _cute_mask_mod(mask_mod)
    if plan.family == 100:
        from fa4_jax.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100

        return FlashAttentionBackwardSm100(
            head_dim,
            head_dim_v,
            is_causal=causal,
            is_local=local,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=plan.tile_m,
            tile_n=plan.tile_n,
            deterministic=False,
            spt=False,
            cluster_size=1,
            use_2cta_instrs=False,
            mask_mod=mm,
            has_aux_tensors=n_aux > 0,
        )
    if plan.family == 90:
        from fa4_jax.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90

        return FlashAttentionBackwardSm90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            **_sm90_bwd_kwargs(plan.sm90, causal, local),
            mask_mod=mm,
            has_aux_tensors=n_aux > 0,
        )
    c = plan.sm80
    return _bwd_kernel_cls(arch)(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        c.tile_m,
        c.tile_n,
        c.num_stages,  # num_stages_Q
        c.num_stages,  # num_stages_dO
        c.num_threads,
        False,  # pack_gqa
        causal,
        local,
        False,  # SdP_swapAB
        False,  # dKV_swapAB
        False,  # dQ_swapAB
        c.AtomLayoutMSdP,
        c.AtomLayoutNdKV,
        c.AtomLayoutMdQ,
        V_in_regs=c.V_in_regs,
        mask_mod=mm,
        has_aux_tensors=n_aux > 0,
    )


@functools.cache
def _get_bwd_pre(dtype, head_dim, head_dim_v, tile_m):
    from fa4_jax.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess

    return FlashAttentionBackwardPreprocess(dtype, head_dim, head_dim_v, tile_m)


def _bwd_post_params(plan: _BwdPlan, arch: int, which: str) -> tuple:
    """(layout_arch, block, num_threads, atom_layout, swap_ab) of the
    postprocess that converts the accumulator ``which`` ("dq" or "dkv").

    These select the accumulator element layout, which must match the kernel
    that wrote it, not the device: the tcgen05 backward writes the SM100
    layout (AtomLayoutMdQ=1, upstream's default), the wgmma backward its
    config row's MMA layouts -- different for dQ (tile_m rows, AtomLayoutMdQ,
    dQ_swapAB, and one warpgroup when dQ_single_wg) and dK/dV (tile_n rows,
    AtomLayoutNdKV, dKV_swapAB) -- and the SM80-MMA kernel the sm_8x/12x one,
    including when it runs on sm_90/sm_100 as the fallback.
    """
    if plan.family == 100:
        return arch, plan.tile_m, _BWD_NUM_THREADS, 1, False
    if plan.family == 90:
        c = plan.sm90
        if which == "dq":
            threads = 128 if c.dQ_single_wg else c.num_wg * 128
            return 90, c.tile_m, threads, c.AtomLayoutMdQ, c.dQ_swapAB
        return 90, c.tile_n, c.num_wg * 128, c.AtomLayoutNdKV, c.dKV_swapAB
    layout_arch = arch if arch // 10 in (8, 12) else 120
    c = plan.sm80
    if which == "dq":
        return layout_arch, c.tile_m, c.num_threads, c.AtomLayoutMdQ, False
    return layout_arch, c.tile_n, c.num_threads, c.AtomLayoutNdKV, False


@functools.cache
def _get_bwd_post(dtype, head_dim, layout_arch, block, num_threads, atom_layout, swap_ab):
    """Postprocess for one accumulator; head_dim is the converted tensor's
    (dQ/dK take the problem's head_dim, dV takes head_dim_v). The layout
    parameters come from _bwd_post_params."""
    from fa4_jax.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess

    return FlashAttentionBackwardPostprocess(
        dtype, head_dim, layout_arch, block, num_threads, atom_layout, swap_ab
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
    """dQaccum -> dQ."""
    tree["post"](mdQaccum, mdQ, cutlass.Float32(tree["scale"]), None, None, None, stream=stream)


@cute.jit
def _fa4_bwd_post_dk(stream, tree, mdKaccum, mdK):
    """dKaccum -> dK under GQA: same head_dim and scale as dQ, but the kv
    tile's accumulator layout (they differ on sm_90)."""
    tree["post_k"](mdKaccum, mdK, cutlass.Float32(tree["scale"]), None, None, None, stream=stream)


@cute.jit
def _fa4_bwd_post_dv(stream, tree, mdVaccum, mdV):
    """dVaccum -> dV under GQA: head_dim_v tiles, and no softmax scale."""
    tree["post_v"](mdVaccum, mdV, cutlass.Float32(1.0), None, None, None, stream=stream)


# The varlen backward chain: same kernels, packed 3D tensors, and the
# cu_seqlens/seqused arrays as explicit positional operands (the accumulators
# use tile-padded per-sequence offsets, see `_vjp_varlen` for the sizing).


@cute.jit
def _fa4_bwd_pre_varlen(stream, tree, mO, mdO, mLSE, cu_q, su_q, mPdPsum, mLSElog2, mdQaccum):
    tree["pre"](
        mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum,
        cu_q, su_q, None, None, None,  # cu_seqlens_q, seqused_q, dLSE, row_max, scale_p
        cutlass.Float32(tree["scale"]),
        stream=stream,
    )


def _bwd_varlen_body(
    stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum, cu_q, cu_k, su_q, su_k, aux,
    mdQaccum, mdK, mdV,
):
    from fa4_jax.cute.utils import AuxData

    wl, wr = _windows(tree)
    tree["bwd"](
        mQ, mK, mV, mdO, mLSElog2, mPdPsum, mdQaccum, mdK, mdV,
        cutlass.Float32(tree["scale"]),
        cu_q, cu_k, su_q, su_k,
        window_size_left=wl,
        window_size_right=wr,
        aux_data=AuxData(tensors=aux) if cutlass.const_expr(len(aux) > 0) else AuxData(),
        stream=stream,
    )


@cute.jit
def _fa4_bwd_varlen(
    stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum, _mdQaccum_in,
    cu_q, cu_k, su_q, su_k, aux,
    mdQaccum, mdK, mdV,
):
    _bwd_varlen_body(
        stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum,
        cu_q, cu_k, su_q, su_k, aux, mdQaccum, mdK, mdV,
    )


@cute.jit
def _fa4_bwd_varlen_dp(
    stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum, _mdQaccum_in, _mdK_in, _mdV_in,
    cu_q, cu_k, su_q, su_k, aux,
    mdQaccum, mdK, mdV,
):
    """defined_padding variant: dK/dV arrive zero-filled and donated, so the
    kernel's row-predicated stores leave unused rows at exactly 0."""
    _bwd_varlen_body(
        stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum,
        cu_q, cu_k, su_q, su_k, aux, mdQaccum, mdK, mdV,
    )


@cute.jit
def _fa4_bwd_gqa_varlen(
    stream, tree, mQ, mK, mV, mdO, mLSElog2, mPdPsum, _mdQaccum_in, _mdKaccum_in,
    _mdVaccum_in, cu_q, cu_k, su_q, su_k, aux, mdQaccum, mdKaccum, mdVaccum,
):
    from fa4_jax.cute.utils import AuxData

    wl, wr = _windows(tree)
    tree["bwd"](
        mQ, mK, mV, mdO, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum,
        cutlass.Float32(tree["scale"]),
        cu_q, cu_k, su_q, su_k,
        window_size_left=wl,
        window_size_right=wr,
        aux_data=AuxData(tensors=aux) if cutlass.const_expr(len(aux) > 0) else AuxData(),
        stream=stream,
    )


@cute.jit
def _fa4_bwd_post_varlen(stream, tree, mdAccum, cu_q, su_q, mdQ):
    """dQaccum -> dQ."""
    tree["post"](mdAccum, mdQ, cutlass.Float32(tree["scale"]), cu_q, su_q, None, stream=stream)


@cute.jit
def _fa4_bwd_post_varlen_dp(stream, tree, mdAccum, cu_q, su_q, _mdQ_in, mdQ):
    """defined_padding variant: mdQ arrives zero-filled and donated."""
    tree["post"](mdAccum, mdQ, cutlass.Float32(tree["scale"]), cu_q, su_q, None, stream=stream)


@cute.jit
def _fa4_bwd_post_dk_varlen(stream, tree, mdKaccum, cu_k, su_k, mdK):
    """dKaccum -> dK under GQA (k-side seqlen arrays, kv-tile accumulator layout)."""
    tree["post_k"](mdKaccum, mdK, cutlass.Float32(tree["scale"]), cu_k, su_k, None, stream=stream)


@cute.jit
def _fa4_bwd_post_dk_varlen_dp(stream, tree, mdKaccum, cu_k, su_k, _mdK_in, mdK):
    tree["post_k"](mdKaccum, mdK, cutlass.Float32(tree["scale"]), cu_k, su_k, None, stream=stream)


@cute.jit
def _fa4_bwd_post_dv_varlen(stream, tree, mdVaccum, cu_k, su_k, mdV):
    tree["post_v"](mdVaccum, mdV, cutlass.Float32(1.0), cu_k, su_k, None, stream=stream)


@cute.jit
def _fa4_bwd_post_dv_varlen_dp(stream, tree, mdVaccum, cu_k, su_k, _mdV_in, mdV):
    tree["post_v"](mdVaccum, mdV, cutlass.Float32(1.0), cu_k, su_k, None, stream=stream)


# ---------------------------------------------------------------- jvp tangent


@functools.cache
def _get_jvp(dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, mask_mod, n_aux):
    # No arch key: the one SM80-style tangent kernel serves every device.
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
        mCuSeqlensQ=tree.get("cu_q"), mCuSeqlensK=tree.get("cu_k"),
        mSeqUsedQ=tree.get("su_q"), mSeqUsedK=tree.get("su_k"),
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
            causal, local, mask_mod, cfg.n_aux, arch, cfg.varlen,
        )
    elif kind == "bwd":
        plan = _bwd_plan(
            dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, arch
        )
        assert plan is not None, "unreachable: _validate checked _can_implement"
        st["bwd"] = _get_bwd(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            causal, local, mask_mod, cfg.n_aux, arch, plan,
        )
        st["pre"] = _get_bwd_pre(dtype, head_dim, head_dim_v, plan.tile_m)
        st["post"] = _get_bwd_post(dtype, head_dim, *_bwd_post_params(plan, arch, "dq"))
        if qhead_per_kvhead > 1:
            dkv = _bwd_post_params(plan, arch, "dkv")
            st["post_k"] = _get_bwd_post(dtype, head_dim, *dkv)
            st["post_v"] = _get_bwd_post(dtype, head_dim_v, *dkv)
    elif kind == "jvp":
        st["jvp"] = _get_jvp(
            dtype, head_dim, head_dim_v, qhead_per_kvhead,
            causal, local, mask_mod, cfg.n_aux,
        )
    return st


@functools.cache
def _aux_static(name: str, kernel, scale: float | None) -> _StaticDict:
    """The static tree for one backward preprocess/postprocess call: just
    the kernel object (under the key its jit function reads) and the softmax
    scale it bakes in. These kernels depend only on dtype, head dims and tile
    geometry, so handing them the variant's whole static dict -- the main
    backward kernel, windows, mask -- made cutejax rebuild them for every
    causal/window/mask variant (276 of the suite's 709 builds; 94 after).
    Memoized, and the kernel objects are memoized by their parameters, so
    every variant sharing a pre/post kernel shares one compile."""
    st = _StaticDict({name: kernel})
    if scale is not None:
        st["scale"] = scale
    return st


@functools.cache
def _can_implement(
    kind: str, dtype, head_dim: int, head_dim_v: int, qhead_per_kvhead: int,
    causal: bool, local: bool, arch: int,
) -> bool:
    """Ask the kernel class whether our tile config fits (SMEM budget, head
    dim limits); a launch that overflows SMEM would otherwise fail *inside*
    CUDA and leave the context in an error state.

    The budget is always passed explicitly: the SM80-MMA classes default to
    sm_80's 163 KB, which is wrong for every consumer part (sm_86/89/120
    have 99 KB)."""
    import cutlass.utils as utils_basic

    smem_capacity = utils_basic.get_smem_capacity_in_bytes(f"sm_{arch}")
    if kind == "fwd":
        if arch // 10 == 10:
            # The tcgen05 forward has no can_implement; keep the family's
            # public envelope (head dims <= 128 -- upstream's "standard
            # range" for SM100, larger needs 2-CTA MMAs we don't launch).
            return head_dim <= 128 and head_dim_v <= 128
        if arch // 10 == 9:
            from fa4_jax.cute.flash_fwd_sm90 import FlashAttentionForwardSm90

            c = _fwd_cfg_sm90(head_dim, head_dim_v, causal, local)
            return FlashAttentionForwardSm90.can_implement(
                dtype, head_dim, head_dim_v, c.tile_m, c.tile_n, 2,
                (c.tile_m // 64 + 1) * 128, causal,
                Q_in_regs=False, mma_pv_is_rs=c.mma_pv_is_rs, smem_capacity=smem_capacity,
            )
        tile_m, tile_n = _fwd_tile_mn(arch, dtype, head_dim, head_dim_v)
        return _fwd_kernel_cls(arch).can_implement(
            dtype, head_dim, head_dim_v, tile_m, tile_n, 1, 128, causal,
            smem_capacity=smem_capacity,
        )
    if kind == "bwd":
        plan = _bwd_plan(dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, local, arch)
        return plan is not None
    if kind == "jvp":
        from fa4_jax.cute.flash_jvp import FlashAttentionJvpSm80

        tile_m = 128 if max(head_dim, head_dim_v) <= 64 else 64
        return FlashAttentionJvpSm80.can_implement(dtype, head_dim, head_dim_v, tile_m, 64, 128)
    raise ValueError(kind)


def _validate(cfg: AttnConfig, kind: str, q, k, v):
    # Negative indexing serves both layouts: (batch, seqlen, heads, head_dim)
    # and packed varlen (total_tokens, heads, head_dim).
    dtype = _DTYPES.get(jnp.dtype(q.dtype))
    if dtype is None:
        raise ValueError(f"FA4 kernels take fp16/bf16 only, got {q.dtype}")
    head_dim, head_dim_v = q.shape[-1], v.shape[-1]
    if head_dim % 8 or head_dim_v % 8:
        raise ValueError(f"head dims must be multiples of 8, got {head_dim}, {head_dim_v}")
    # Each device runs its own kernel family: SM80-MMA on sm_8x/sm_12x, wgmma
    # on sm_90, tcgen05 on sm_10x (the latter two fall back to the SM80-MMA
    # backward for configs their native kernels can't take; see _bwd_plan).
    arch = _device_arch()
    if arch // 10 not in (8, 9, 10, 11, 12):
        raise ValueError(f"the FA4 kernels need sm_80 or newer, got sm_{arch}")
    ratio = q.shape[-2] // k.shape[-2]
    if not _can_implement(
        kind, dtype, head_dim, head_dim_v, ratio, cfg.causal, cfg.local, arch
    ):
        raise ValueError(
            f"FA4 {kind} kernel cannot run head_dim={head_dim}, head_dim_v={head_dim_v} "
            f"({q.dtype}) on sm_{arch}: tile does not fit the shared-memory budget "
            "or the head dims exceed what the kernel supports (<= 256; <= 128 "
            "on sm_10x and for forward-mode/JVP)"
        )
    return dtype, head_dim, head_dim_v, ratio, arch


class CuteBackend:
    def _tree(self, cfg, kind, q, k, v, aux, seqlens=None, **extra):
        dtype, hd, hdv, ratio, arch = _validate(cfg, kind, q, k, v)
        tree = dict(_get_static(cfg, dtype, hd, hdv, ratio, arch, kind))
        tree.update(q=q, k=k, v=v, **extra)
        tree.update({f"aux{i}": a for i, a in enumerate(aux)})
        spec = _QKV_SPEC if seqlens is None else _QKV3_SPEC
        specs = {"q": spec, "k": spec, "v": spec}
        if seqlens is not None:
            cu_q, cu_k, su_q, su_k = seqlens
            tree.update(cu_q=cu_q, cu_k=cu_k, su_q=su_q, su_k=su_k)
        return tree, specs

    @staticmethod
    def _in_specs(tree, specs):
        # cutejax matches in_specs to array leaves in jax.tree.flatten order,
        # which for a dict is sorted-key order; unlisted leaves (aux tensors,
        # cu_seqlens, lse) default to fully dynamic.
        return [specs.get(name) for name in sorted(tree) if _is_dynamic(tree[name])]

    @staticmethod
    def _tree_aliases(tree, **outs):
        """input_output_aliases for named tree entries, resolved against the
        same sorted-key array-leaf order ``_in_specs`` uses."""
        order = [name for name in sorted(tree) if _is_dynamic(tree[name])]
        return {order.index(name): out_idx for name, out_idx in outs.items()}

    @staticmethod
    def _out_struct(q, v):
        return jax.ShapeDtypeStruct((*q.shape[:-1], v.shape[-1]), q.dtype)

    @staticmethod
    def _lse_struct(q, seqlens):
        if seqlens is not None:
            total_q, h, _ = q.shape
            return jax.ShapeDtypeStruct((h, total_q), jnp.float32)
        b, s_q, h, _ = q.shape
        return jax.ShapeDtypeStruct((b, h, s_q), jnp.float32)

    def fwd(self, cfg, q, k, v, aux, seqlens=None):
        tree, specs = self._tree(cfg, "fwd", q, k, v, aux, seqlens)
        aliases = None
        if cfg.defined_padding and seqlens is not None:
            # Donate a zero-filled output: the kernel's stores are
            # row-predicated on the used lengths, so unused rows keep the 0.
            out_struct = self._out_struct(q, v)
            tree["out0"] = jnp.zeros(out_struct.shape, out_struct.dtype)
            specs["out0"] = specs["q"]
            aliases = self._tree_aliases(tree, out0=0)
        return cutejax.call(
            _fa4_fwd,
            self._out_struct(q, v),
            tree,
            in_specs=self._in_specs(tree, specs),
            out_specs=specs["q"],
            input_output_aliases=aliases,
            name="fa4_fwd" if aliases is None else "fa4_fwd_dp",
        )

    def fwd_lse(self, cfg, q, k, v, aux, seqlens=None):
        tree, specs = self._tree(cfg, "fwd", q, k, v, aux, seqlens)
        aliases = None
        if cfg.defined_padding and seqlens is not None:
            out_struct = self._out_struct(q, v)
            lse_struct = self._lse_struct(q, seqlens)
            tree["out0"] = jnp.zeros(out_struct.shape, out_struct.dtype)
            tree["lse0"] = jnp.full(lse_struct.shape, -jnp.inf, lse_struct.dtype)
            specs.update(out0=specs["q"], lse0=_LSE_SPEC)
            aliases = self._tree_aliases(tree, out0=0, lse0=1)
        return cutejax.call(
            _fa4_fwd_lse,
            [self._out_struct(q, v), self._lse_struct(q, seqlens)],
            tree,
            in_specs=self._in_specs(tree, specs),
            out_specs=[specs["q"], _LSE_SPEC],
            input_output_aliases=aliases,
            name="fa4_fwd_lse" if aliases is None else "fa4_fwd_lse_dp",
        )

    def tangent(self, cfg, q, k, v, out, lse, dq, dk, dv, aux, seqlens=None):
        tree, specs = self._tree(
            cfg, "jvp", q, k, v, aux, seqlens, dq=dq, dk=dk, dv=dv, out=out, lse=lse
        )
        qspec = specs["q"]
        specs.update(dq=qspec, dk=qspec, dv=qspec, out=qspec, lse=_LSE_SPEC)
        aliases = None
        if cfg.defined_padding and seqlens is not None:
            # Tangents of the padding are 0 (the tangent of a constant),
            # including the mu/t_lse channel.
            out_struct = self._out_struct(q, v)
            lse_struct = self._lse_struct(q, seqlens)
            tree["tout0"] = jnp.zeros(out_struct.shape, out_struct.dtype)
            tree["tmu0"] = jnp.zeros(lse_struct.shape, lse_struct.dtype)
            specs.update(tout0=qspec, tmu0=_LSE_SPEC)
            aliases = self._tree_aliases(tree, tout0=0, tmu0=1)
        return cutejax.call(
            _fa4_jvp,
            [self._out_struct(q, v), self._lse_struct(q, seqlens)],
            tree,
            in_specs=self._in_specs(tree, specs),
            out_specs=[qspec, _LSE_SPEC],
            input_output_aliases=aliases,
            name="fa4_jvp" if aliases is None else "fa4_jvp_dp",
        )

    def vjp(self, cfg, q, k, v, out, lse, d_out, d_lse, aux, seqlens=None):
        if d_lse is not None:
            raise NotImplementedError(
                "reverse mode through the lse output is not wired to the cute "
                "backward yet; use backend='reference' if you need d(lse)."
            )
        d_out = d_out.astype(q.dtype)
        if seqlens is not None:
            return self._vjp_varlen(cfg, q, k, v, out, lse, d_out, aux, *seqlens)
        b, s_q, h, d = q.shape
        _, s_kv, h_kv, d_v = v.shape
        d_pad = _round_up(d, 32)
        f32 = jnp.float32
        dtype, hd, hdv, ratio, arch = _validate(cfg, "bwd", q, k, v)
        plan = _bwd_plan(dtype, hd, hdv, ratio, cfg.causal, cfg.local, arch)
        s_q_pad = _round_up(s_q, plan.tile_m)
        st = _get_static(cfg, dtype, hd, hdv, ratio, arch, "bwd")

        dpsum, lse_log2, dq_accum = cutejax.call(
            _fa4_bwd_pre,
            [
                jax.ShapeDtypeStruct((b, h, s_q_pad), f32),
                jax.ShapeDtypeStruct((b, h, s_q_pad), f32),
                jax.ShapeDtypeStruct((b, h, s_q_pad * d_pad), f32),
            ],
            _aux_static("pre", st["pre"], st["scale"]), out, d_out, lse,
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
            s_kv_pad = _round_up(s_kv, plan.tile_n)
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
                _fa4_bwd_post_dk,
                jax.ShapeDtypeStruct(k.shape, k.dtype),
                _aux_static("post_k", st["post_k"], st["scale"]), dk_accum,
                in_specs=[_F32_SPEC],
                out_specs=_QKV_SPEC,
                name="fa4_bwd_post_dk",
            )
            dv = cutejax.call(
                _fa4_bwd_post_dv,
                jax.ShapeDtypeStruct(v.shape, v.dtype),
                _aux_static("post_v", st["post_v"], None), dv_accum,
                in_specs=[_F32_SPEC],
                out_specs=_QKV_SPEC,
                name="fa4_bwd_post_dv",
            )

        dq = cutejax.call(
            _fa4_bwd_post,
            jax.ShapeDtypeStruct((b, s_q, h, d), q.dtype),
            _aux_static("post", st["post"], st["scale"]), dq_accum,
            in_specs=[_F32_SPEC],
            out_specs=_QKV_SPEC,
            name="fa4_bwd_post",
        )
        return dq, dk, dv

    def _vjp_varlen(self, cfg, q, k, v, out, lse, d_out, aux, cu_q, cu_k, su_q, su_k):
        """The backward chain over the packed layout.

        Same three kernels as the batched path; what changes is the fp32
        accumulator geometry. The kernels place sequence ``b``'s accumulator
        rows at the tile-rounded offset ``(cu[b] + b*tile) // tile * tile``
        spanning ``round_up(len_b, tile)`` rows, so
        ``round_up(total, tile) + batch * tile`` bounds the packing — and
        keeps the innermost dim a tile multiple, honouring _F32_SPEC's
        vectorization promise. LSE/dpsum lose their batch dim:
        ``(heads, padded_total)``.
        """
        f32 = jnp.float32
        total_q, h, d = q.shape
        total_k, h_kv, _ = k.shape
        d_v = v.shape[2]
        n_batch = cu_q.shape[0] - 1
        d_pad = _round_up(d, 32)
        dtype, hd, hdv, ratio, arch = _validate(cfg, "bwd", q, k, v)
        plan = _bwd_plan(dtype, hd, hdv, ratio, cfg.causal, cfg.local, arch)
        s_q_pad = _round_up(total_q, plan.tile_m) + n_batch * plan.tile_m
        st = _get_static(cfg, dtype, hd, hdv, ratio, arch, "bwd")

        dpsum, lse_log2, dq_accum = cutejax.call(
            _fa4_bwd_pre_varlen,
            [
                jax.ShapeDtypeStruct((h, s_q_pad), f32),
                jax.ShapeDtypeStruct((h, s_q_pad), f32),
                jax.ShapeDtypeStruct((h, s_q_pad * d_pad), f32),
            ],
            _aux_static("pre", st["pre"], st["scale"]), out, d_out, lse, cu_q, su_q,
            in_specs=[_QKV3_SPEC, _QKV3_SPEC, _LSE_SPEC, None, None],
            out_specs=_F32_SPEC,
            name="fa4_bwd_pre_varlen",
        )

        defined = cfg.defined_padding
        if ratio == 1:
            if defined:
                # Donate zero-filled dK/dV: the epilogue's row-predicated
                # stores never touch rows past seqused_k, so they stay 0.
                dq_accum, dk, dv = cutejax.call(
                    _fa4_bwd_varlen_dp,
                    [
                        jax.ShapeDtypeStruct(dq_accum.shape, f32),
                        jax.ShapeDtypeStruct(k.shape, k.dtype),
                        jax.ShapeDtypeStruct(v.shape, v.dtype),
                    ],
                    st, q, k, v, d_out, lse_log2, dpsum, dq_accum,
                    jnp.zeros(k.shape, k.dtype), jnp.zeros(v.shape, v.dtype),
                    cu_q, cu_k, su_q, su_k, tuple(aux),
                    in_specs=[_QKV3_SPEC] * 4 + [_F32_SPEC] * 3 + [_QKV3_SPEC] * 2
                    + [None] * (4 + len(aux)),
                    out_specs=[_F32_SPEC, _QKV3_SPEC, _QKV3_SPEC],
                    input_output_aliases={6: 0, 7: 1, 8: 2},
                    name="fa4_bwd_varlen_dp",
                )
            else:
                dq_accum, dk, dv = cutejax.call(
                    _fa4_bwd_varlen,
                    [
                        jax.ShapeDtypeStruct(dq_accum.shape, f32),
                        jax.ShapeDtypeStruct(k.shape, k.dtype),
                        jax.ShapeDtypeStruct(v.shape, v.dtype),
                    ],
                    st, q, k, v, d_out, lse_log2, dpsum, dq_accum,
                    cu_q, cu_k, su_q, su_k, tuple(aux),
                    in_specs=[_QKV3_SPEC] * 4 + [_F32_SPEC] * 3 + [None] * (4 + len(aux)),
                    out_specs=[_F32_SPEC, _QKV3_SPEC, _QKV3_SPEC],
                    input_output_aliases={6: 0},
                    name="fa4_bwd_varlen",
                )
        else:
            s_kv_pad = _round_up(total_k, plan.tile_n) + n_batch * plan.tile_n
            dk_accum = jnp.zeros((h_kv, s_kv_pad * d_pad), f32)
            dv_accum = jnp.zeros((h_kv, s_kv_pad * _round_up(d_v, 32)), f32)
            dq_accum, dk_accum, dv_accum = cutejax.call(
                _fa4_bwd_gqa_varlen,
                [
                    jax.ShapeDtypeStruct(dq_accum.shape, f32),
                    jax.ShapeDtypeStruct(dk_accum.shape, f32),
                    jax.ShapeDtypeStruct(dv_accum.shape, f32),
                ],
                st, q, k, v, d_out, lse_log2, dpsum, dq_accum, dk_accum, dv_accum,
                cu_q, cu_k, su_q, su_k, tuple(aux),
                in_specs=[_QKV3_SPEC] * 4 + [_F32_SPEC] * 5 + [None] * (4 + len(aux)),
                out_specs=[_F32_SPEC] * 3,
                input_output_aliases={6: 0, 7: 1, 8: 2},
                name="fa4_bwd_gqa_varlen",
            )
            dk = self._post_varlen(
                _fa4_bwd_post_dk_varlen, _fa4_bwd_post_dk_varlen_dp, defined,
                jax.ShapeDtypeStruct(k.shape, k.dtype),
                _aux_static("post_k", st["post_k"], st["scale"]), dk_accum, cu_k, su_k,
                "fa4_bwd_post_dk_varlen",
            )
            dv = self._post_varlen(
                _fa4_bwd_post_dv_varlen, _fa4_bwd_post_dv_varlen_dp, defined,
                jax.ShapeDtypeStruct(v.shape, v.dtype),
                _aux_static("post_v", st["post_v"], None), dv_accum, cu_k, su_k,
                "fa4_bwd_post_dv_varlen",
            )

        dq = self._post_varlen(
            _fa4_bwd_post_varlen, _fa4_bwd_post_varlen_dp, defined,
            jax.ShapeDtypeStruct((total_q, h, d), q.dtype),
            _aux_static("post", st["post"], st["scale"]), dq_accum, cu_q, su_q,
            "fa4_bwd_post_varlen",
        )
        return dq, dk, dv

    @staticmethod
    def _post_varlen(fn, fn_dp, defined, struct, st, accum, cu, su, name):
        """Accumulator -> grad postprocess; with defined_padding the output
        buffer is donated zero-filled so unused rows stay 0."""
        if defined:
            return cutejax.call(
                fn_dp, struct,
                st, accum, cu, su, jnp.zeros(struct.shape, struct.dtype),
                in_specs=[_F32_SPEC, None, None, _QKV3_SPEC],
                out_specs=_QKV3_SPEC,
                input_output_aliases={3: 0},
                name=name + "_dp",
            )
        return cutejax.call(
            fn, struct,
            st, accum, cu, su,
            in_specs=[_F32_SPEC, None, None],
            out_specs=_QKV3_SPEC,
            name=name,
        )


def _is_dynamic(leaf) -> bool:
    """Whether a tree leaf is a jax array (dynamic data) rather than static."""
    return isinstance(leaf, (jax.Array, jax.core.Tracer))


register_backend("cute", CuteBackend())
