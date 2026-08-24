# fa4-jax original (not upstream flash-attention); BSD-3-Clause like the rest
# of this directory. Modeled closely on flash_fwd.py's FlashAttentionForwardSm80.
"""Forward-mode (JVP) tangent kernel for FlashAttention-4, SM80/SM120 family.

Given the forward's output and LSE as residuals, the attention JVP for
tangents (dQ, dK, dV) is

    P     = exp(S * scale - lse)                  (exact, no online softmax)
    tS    = scale * (dQ K^T + Q dK^T)
    mu    = rowsum(P * tS)                        (= the tangent of lse)
    t_out = (P * tS) V + P dV - mu * out

so one pass over the KV sequence with five GEMMs per tile computes the full
tangent: S = Q K^T, the two tS terms, (P*tS) V, and P dV. Because lse is
known, P needs no running max/sum bookkeeping — each tile's probabilities are
final on first computation, and `mu` is a plain row accumulator, quad-reduced
once at the end. The epilogue subtracts `mu * out` (out re-read from gmem)
and then reuses the forward kernel's epilogue verbatim, writing t_out where
it would write O and mu where it would write LSE.

Masking (seqlen tails, causal, local windows, and flex-attention mask mods
with aux tensors) reuses `AttentionMask` exactly as the forward does: masked
scores go to -inf, so their P (and hence every tangent term) vanishes.
Varlen (packed layouts with cu_seqlens/seqused) follows the forward kernel
verbatim: per-batch pointer offsets from `SeqlenInfoQK`, the varlen tile
scheduler, and the shared (fully predicated, non-TMA) epilogue. Rows
with no visible key have lse = -inf; those are mapped to P = 0 rather than
NaN, giving t_out = 0 to match the forward's zero-output convention. GQA is
supported the same way as the forward (kv-head = q-head // ratio) — unlike
the backward, nothing here accumulates into KV-shaped outputs.

Single-stage pipeline only (the six input tiles leave no smem for double
buffering): cp.async loads of the next K/dK overlap the softmax-and-PV phase,
and V/dV loads overlap the S GEMMs, mirroring flash_fwd's num_stages=1
schedule.
"""

import math
from types import SimpleNamespace
import operator
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import layout_utils

from fa4_jax.cute import ampere_helpers as sm80_utils
from fa4_jax.cute import utils
from fa4_jax.cute.cute_dsl_utils import assume_tensor_aligned
from fa4_jax.cute.flash_fwd import FlashAttentionForwardSm80
from fa4_jax.cute.mask import AttentionMask
from fa4_jax.cute.seqlen_info import SeqlenInfoQK
from fa4_jax.cute.block_info import BlockInfo
from fa4_jax.cute.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from fa4_jax.cute.utils import AuxData

LOG2_E = math.log2(math.e)


class FlashAttentionJvpSm80(FlashAttentionForwardSm80):
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        tile_m: int = 128,
        tile_n: int = 64,
        num_threads: int = 128,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
    ):
        super().__init__(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=is_causal,
            is_local=is_local,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=1,
            num_threads=num_threads,
            Q_in_regs=False,
            score_mod=None,
            mask_mod=mask_mod,
            has_aux_tensors=has_aux_tensors,
        )

    @staticmethod
    def can_implement(dtype, head_dim, head_dim_v, tile_m, tile_n, num_threads) -> bool:
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0 or head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0 or num_threads % 32 != 0 or (tile_m * 2) % num_threads != 0:
            return False
        # The epilogue reuses the sQ+sdQ region for the t_out tile.
        if head_dim_v > 2 * head_dim:
            return False
        # Q-like tiles twice, K-like twice, V-like twice; consumer parts have
        # ~99KB usable per block.
        smem = 2 * (tile_m * head_dim + tile_n * head_dim + tile_n * head_dim_v) * (dtype.width // 8)
        return smem <= 99 * 1024

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]

        # sQ and sdQ sit last and adjacent: the epilogue's sTO tile
        # (tile_m x tile_hdimv) aliases sQ and may spill into sdQ, which
        # can_implement bounds by head_dim_v <= 2 * head_dim.
        @cute.struct
        class SharedStorageJvp:
            sV: sV_struct
            sdV: sV_struct
            sK: sK_struct
            sdK: sK_struct
            sQ: sQ_struct
            sdQ: sQ_struct

        return SharedStorageJvp

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mTO: cute.Tensor,
        mMu: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        aux_data: AuxData = AuxData(),
        # Always keep stream as the last parameter (EnvStream via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Launch the tangent kernel.

        All of mQ/mK/mV/mdQ/mdK/mdV/mO/mTO are (batch, seqlen, heads,
        head_dim) in the same fp16/bf16 dtype; mLSE/mMu are (batch, heads,
        seqlen_q) fp32. mMu receives the LSE tangent. With mCuSeqlensQ/K the
        layouts are packed varlen — (total, heads, head_dim) and
        (heads, total_q) — with optional mSeqUsedQ/K used-length overrides,
        exactly as in the forward kernel.
        """
        for t in (mK, mV, mdQ, mdK, mdV, mO, mTO):
            if const_expr(t.element_type != mQ.element_type):
                raise TypeError("q/k/v/dq/dk/dv/out/t_out must share one dtype")
        if const_expr(mLSE.element_type != Float32 or mMu.element_type != Float32):
            raise TypeError("lse/mu must be Float32")
        for t in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK):
            if const_expr(t is not None and t.element_type != Int32):
                raise TypeError("cu_seqlens/seqused tensors must be Int32")
        assert mQ.element_type == self.dtype

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        self.use_tma_O = False
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mdQ, mdK, mdV, mO, mTO = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mdQ, mdK, mdV, mO, mTO)
        ]
        # (batch, seqlen, head, head_dim) -> (seqlen, head_dim, head, batch),
        # or packed varlen (total, head, head_dim) -> (total, head_dim, head)
        qo_t = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        kv_t = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        lse_t = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mQ, mdQ, mO, mTO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=qo_t))
            for t in (mQ, mdQ, mO, mTO)
        ]
        mK, mV, mdK, mdV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=kv_t))
            for t in (mK, mV, mdK, mdV)
        ]
        mLSE, mMu = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=lse_t)) for t in (mLSE, mMu)
        ]

        # Varlen tiles are scheduled exactly as in the forward: per-batch
        # m-block counts decoded from cu_seqlens/seqused by the varlen
        # scheduler; wasted grid tiles resolve to empty seqlens and fall
        # through every predicated load/store.
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_batch = (
            mCuSeqlensQ.shape[0] - 1
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[3])
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=1,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, _ = utils.compute_softmax_scale_log2(softmax_scale, None)
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, 1, False, aux_data.tensors)

        self.kernel(
            mQ,
            mK,
            mV,
            mdQ,
            mdK,
            mdV,
            mO,
            mLSE,
            mTO,
            mMu,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            aux_data,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mTO: cute.Tensor,
        mMu: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        tidx, _, _ = cute.arch.thread_idx()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, num_head, batch_size, _ = work_tile.tile_idx

        # fa4-jax fix: the varlen scheduler's grid is an upper bound, and wasted
        # tiles decode to batch_idx == num_batch -- for which SeqlenInfoQK would
        # read cu_seqlens[num_batch + 1] / seqused[num_batch] out of bounds and
        # act on garbage seqlens (observed as illegal-address faults). Skip them
        # entirely, as the backward kernels do; the condition is CTA-uniform so
        # the barriers inside stay safe.
        if work_tile.is_valid_tile:
            block_info = BlockInfo(
                self.tile_m,
                self.tile_n,
                self.is_causal,
                self.is_local,
                False,  # is_split_kv
                window_size_left,
                window_size_right,
                qhead_per_kvhead_packgqa=1,
            )
            seqlen = SeqlenInfoQK.create(
                batch_idx=batch_size,
                seqlen_q_static=mQ.shape[0],
                seqlen_k_static=mK.shape[0],
                mCuSeqlensQ=mCuSeqlensQ,
                mCuSeqlensK=mCuSeqlensK,
                mSeqUsedQ=mSeqUsedQ,
                mSeqUsedK=mSeqUsedK,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            n_block = cutlass.max(n_block_max - 1, 0)

            # ------------------------------------------------ gmem tiles
            blkQ_shape = (self.tile_m, self.tile_hdim)
            blkK_shape = (self.tile_n, self.tile_hdim)
            blkV_shape = (self.tile_n, self.tile_hdimv)
            num_head_kv = num_head // self.qhead_per_kvhead
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_size, dim=3)[None, None, num_head]
            mdQ_cur = seqlen.offset_batch_Q(mdQ, batch_size, dim=3)[None, None, num_head]
            mO_cur = seqlen.offset_batch_Q(mO, batch_size, dim=3)[None, None, num_head]
            if const_expr(not seqlen.has_cu_seqlens_k):
                mK_cur = mK[None, None, num_head_kv, batch_size]
                mV_cur = mV[None, None, num_head_kv, batch_size]
                mdK_cur = mdK[None, None, num_head_kv, batch_size]
                mdV_cur = mdV[None, None, num_head_kv, batch_size]
            else:
                mK_cur, mV_cur, mdK_cur, mdV_cur = [
                    cute.domain_offset((seqlen.offset_k, 0), t[None, None, num_head_kv])
                    for t in (mK, mV, mdK, mdV)
                ]
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_size, dim=2)[None, num_head]
            gQ = cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
            gdQ = cute.local_tile(mdQ_cur, blkQ_shape, (m_block, 0))
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
            gdK = cute.local_tile(mdK_cur, blkK_shape, (None, 0))
            gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))
            gdV = cute.local_tile(mdV_cur, blkV_shape, (None, 0))

            # ------------------------------------------------ smem
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sdQ = storage.sdQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            sdK = storage.sdK.get_tensor(sK_layout)
            sV = storage.sV.get_tensor(sV_layout)
            sdV = storage.sdV.get_tensor(sV_layout)
            sVt = layout_utils.transpose_view(sV)
            sdVt = layout_utils.transpose_view(sdV)

            gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
            gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
            tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
            tdKsdK, tdKgdK = gmem_thr_copy_K.partition_D(sdK), gmem_thr_copy_K.partition_S(gdK)
            tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)
            tdVsdV, tdVgdV = gmem_thr_copy_V.partition_D(sdV), gmem_thr_copy_V.partition_S(gdV)

            # ------------------------------------------------ mma partitions
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            thr_mma_pv = tiled_mma_pv.get_slice(tidx)
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
            tSrdQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sdQ))
            tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
            tSrdK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sdK[None, None, 0]))
            tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
            tOrdVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sdVt[None, None, 0]))
            acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
            acc_TO = cute.make_rmem_tensor(acc_shape_O, Float32)
            acc_TO.fill(0.0)

            # ------------------------------------------------ smem->rmem copies
            smem_copy_atom_QK = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
            )
            smem_copy_atom_V = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
            )
            smem_thr_copy_Q = utils.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_K = utils.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
            smem_thr_copy_V = utils.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSsdQ = smem_thr_copy_Q.partition_S(sdQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSsdK = smem_thr_copy_K.partition_S(sdK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOsdVt = smem_thr_copy_V.partition_S(sdVt)

            # ------------------------------------------------ load predicates
            cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tKcK = gmem_thr_copy_K.partition_S(cK)
            t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
            if const_expr(self.tile_hdim == self.tile_hdimv):
                tVcV, t0VcV = tKcK, t0KcK
            else:
                cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
                tVcV = gmem_thr_copy_V.partition_S(cV)
                t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
            tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
            if const_expr(self.same_hdim_kv):
                tVpV = tKpK
            else:
                tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])

            load_K = partial(
                self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK, seqlen=seqlen.seqlen_k
            )
            load_dK = partial(
                self.load_K, gmem_tiled_copy_K, tdKgdK, tdKsdK, tKcK, t0KcK, tKpK, seqlen=seqlen.seqlen_k
            )
            load_V = partial(
                self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV, seqlen=seqlen.seqlen_k
            )
            load_dV = partial(
                self.load_V, gmem_tiled_copy_V, tdVgdV, tdVsdV, tVcV, t0VcV, tVpV, seqlen=seqlen.seqlen_k
            )

            # ------------------------------------------------ per-row state
            # Row bookkeeping matches Softmax.create in the forward: each thread
            # owns num_rows rows of the C tile, spread over quads.
            num_rows = acc_TO.shape[0][0] * acc_TO.shape[1]
            cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
            # lse (natural log) -> log2 units once; +inf where the row is out of
            # range or fully masked (lse = -inf), which zeroes P instead of NaN.
            lse_log2 = cute.make_rmem_tensor(num_rows, Float32)
            for r in cutlass.range_constexpr(num_rows):
                row = tScS_mn[r, 0][0] + m_block * self.tile_m
                lse_val = Float32.inf
                # guarded load, not a select: the out-of-range address must never
                # be dereferenced
                if row < seqlen.seqlen_q:
                    lse_val = mLSE_cur[row] * LOG2_E
                # fully-masked rows have lse = -inf; +inf makes P = 0, not NaN
                if lse_val == -Float32.inf:
                    lse_val = Float32.inf
                lse_log2[r] = lse_val
            mu = cute.make_rmem_tensor(num_rows, Float32)
            mu.fill(0.0)

            mask = AttentionMask(
                self.tile_m,
                self.tile_n,
                seqlen,
                window_size_left,
                window_size_right,
                1,  # qhead_per_kvhead_packgqa
            )
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_size,
                head_idx=num_head,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
            )

            # State is threaded through explicit partial-bound arguments, exactly
            # like the forward kernel: the DSL rejects closures over mutable
            # values inside staged loops.
            mma_params = SimpleNamespace(
                thr_mma_qk=thr_mma_qk,
                thr_mma_pv=thr_mma_pv,
                tSrQ=tSrQ,
                tSrdQ=tSrdQ,
                tSrK=tSrK,
                tSrdK=tSrdK,
                tOrVt=tOrVt,
                tOrdVt=tOrdVt,
                acc_TO=acc_TO,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_Q=smem_thr_copy_Q,
                smem_thr_copy_K=smem_thr_copy_K,
                smem_thr_copy_V=smem_thr_copy_V,
                tSsQ=tSsQ,
                tSsdQ=tSsdQ,
                tSsK=tSsK,
                tSsdK=tSsdK,
                tOsVt=tOsVt,
                tOsdVt=tOsdVt,
            )
            row_params = SimpleNamespace(mu=mu, lse_log2=lse_log2)
            compute_one_n_block = partial(
                self.compute_one_n_block,
                mma_params=mma_params,
                smem_copy_params=smem_copy_params,
                row_params=row_params,
                load_K=load_K,
                load_dK=load_dK,
                load_V=load_V,
                load_dV=load_dV,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
            )

            # ------------------------------------------------ prologue
            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
            self.load_Q(gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
            self.load_Q(gmem_thr_copy_Q, gdQ, sdQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1])
            cute.arch.cp_async_commit_group()
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            load_dK(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()

            # ------------------------------------------------ mainloop
            # Same phase split as the forward: the last block masks the seqlen
            # tail; near the causal/local boundary the band masks apply; interior
            # blocks skip masking entirely (a mask_mod always applies).
            compute_one_n_block(
                n_block,
                mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                is_first_n_block=True,
            )
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_local_mask, unroll=1):
                    compute_one_n_block(
                        n_block_max - 2 - n_tile,
                        mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                    )
                n_block = n_block_min_causal_local_mask
            # Interior blocks: mask_seqlen off, but the mask_fn still applies —
            # a local window's left edge and any mask_mod bite here (for pure
            # causal it is a numeric no-op, same as upstream flash_fwd).
            for n_tile in cutlass.range(cutlass.max(n_block - n_block_min, 0), unroll=1):
                compute_one_n_block(
                    n_block - n_tile - 1,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False)
                    if const_expr(self.is_local or self.mask_mod is not None)
                    else None,
                )

            # ------------------------------------------------ finalize
            # mu rows live across 4 threads (quads of the mma C layout).
            mu.store(utils.warp_reduce(mu.load(), operator.add, width=4))

            # t_out -= mu * out, reading out straight from gmem through the mma's
            # C partitioning (predicated on the seqlen tail and the head_dim pad).
            acc_TO_mn = layout_utils.reshape_acc_to_mn(acc_TO)
            cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
            taccOcO_mn = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cO))
            taccOgO_mn = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(gO))
            for r in cutlass.range_constexpr(cute.size(acc_TO_mn.shape[0])):
                if taccOcO_mn[r, 0][0] < seqlen.seqlen_q - m_block * self.tile_m:
                    for c in cutlass.range_constexpr(cute.size(acc_TO_mn.shape[1])):
                        if taccOcO_mn[r, c][1] < mO.shape[1]:
                            acc_TO_mn[r, c] = (
                                acc_TO_mn[r, c] - mu[r] * Float32(taccOgO_mn[r, c])
                            )

            # ------------------------------------------------ epilogue
            # The forward epilogue writes acc -> mTO and rows -> mMu; exactly the
            # store path we need, with mu standing in for the lse.
            sO = cute.make_tensor(sQ.iterator, sO_layout)
            self.epilogue(
                acc_TO,
                mu,
                mTO,
                mMu,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                None,
                tiled_mma_pv,
                tidx,
                m_block,
                num_head,
                batch_size,
            )

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
    ):
        """K/dK tile load with the seqlen tail *zero-filled*.

        The forward's load_K skips the out-of-range rows and leaves stale
        shared memory there ("we'll mask out the scores anyway"): fine for S,
        whose tail columns are overwritten with -inf. The tangent kernel also
        forms tS = dQ K^T + Q dK^T from those rows, and then P * tS with
        P = 0 — which is NaN whenever the stale smem held inf/NaN (e.g. from
        a preceding masked-softmax XLA kernel). Same predicate-as-zfill
        scheme the forward already uses for V, where P @ V has the identical
        hazard.
        """
        is_even_n_smem_k = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        stage = smem_pipe_write if const_expr(self.num_stages > 1) else 0
        if const_expr(need_predicates or not is_even_n_smem_k):
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                # If tile_n doesn't evenly divide the tiled copy, only the last `n` needs checking
                if (
                    is_even_n_smem_k
                    or n < cute.size(tKsK.shape[1]) - 1
                    or tKcK[0, n, 0][0] < self.tile_n
                ):
                    predicate = tKpK[None, n, None] if const_expr(self.check_hdim_oob) else None
                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.tile_n - tKcK[0][0]
                        predicate_n = t0KcK[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tKpK[None, 0, None])
                        for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                                predicate[i, k] = (
                                    tKpK[i, n, k] if const_expr(self.check_hdim_oob) else True
                                ) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tKgK[None, n, None, block],
                        tKsK[None, n, None, stage],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tKgK[None, None, None, block],
                tKsK[None, None, None, stage],
                pred=tKpK if const_expr(self.check_hdim_oob) else None,
            )

    @cute.jit
    def compute_one_n_block(
        self,
        n_blk: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        row_params: SimpleNamespace,
        load_K: Callable,
        load_dK: Callable,
        load_V: Callable,
        load_dV: Callable,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        """One KV tile of the tangent: S/tS GEMMs, exact P, mu, PV GEMMs."""

        def sync():
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

        acc_S = cute.make_rmem_tensor(
            mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n)), Float32
        )
        acc_tS = cute.make_rmem_tensor(acc_S.shape, Float32)
        acc_S.fill(0.0)
        acc_tS.fill(0.0)
        # Q/dQ (prologue) and K/dK (previous iteration) are in smem.
        sync()
        # V/dV for this block: their loads overlap the S/tS GEMMs below.
        load_V(n_blk, smem_pipe_write=0, need_predicates=is_first_n_block)
        load_dV(n_blk, smem_pipe_write=0, need_predicates=is_first_n_block)
        cute.arch.cp_async_commit_group()
        sm80_utils.gemm(
            mma_params.thr_mma_qk, acc_S, mma_params.tSrQ, mma_params.tSrK,
            smem_copy_params.tSsQ, smem_copy_params.tSsK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
        )
        sm80_utils.gemm(
            mma_params.thr_mma_qk, acc_tS, mma_params.tSrdQ, mma_params.tSrK,
            smem_copy_params.tSsdQ, smem_copy_params.tSsK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
        )
        sm80_utils.gemm(
            mma_params.thr_mma_qk, acc_tS, mma_params.tSrQ, mma_params.tSrdK,
            smem_copy_params.tSsQ, smem_copy_params.tSsdK[None, None, None, 0],
            smem_copy_params.smem_thr_copy_Q, smem_copy_params.smem_thr_copy_K,
        )
        # V/dV arrived; every thread is done reading sK/sdK.
        sync()
        if n_blk - 1 >= 0:
            load_K(n_blk - 1, smem_pipe_write=0, need_predicates=False)
            load_dK(n_blk - 1, smem_pipe_write=0, need_predicates=False)
        cute.arch.cp_async_commit_group()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_blk)
        # P = exp2(S*scale*log2e - lse*log2e); pts = P * (tS * scale); both
        # written back over the fp32 accumulators, then converted for the MMA.
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        acc_tS_mn = layout_utils.reshape_acc_to_mn(acc_tS)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn.shape[0])):
            p_row = cute.math.exp2(
                acc_S_mn[r, None].load() * softmax_scale_log2 - row_params.lse_log2[r],
                fastmath=True,
            )
            pts_row = p_row * (acc_tS_mn[r, None].load() * softmax_scale)
            row_params.mu[r] = utils.fadd_reduce(pts_row, init_val=row_params.mu[r], arch=80)
            acc_S_mn[r, None].store(p_row)
            acc_tS_mn[r, None].store(pts_row)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        rPtS = cute.make_fragment_like(acc_tS, self.dtype)
        rPtS.store(acc_tS.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        tOrPtS = layout_utils.reshape_acc_to_frgA(rPtS)
        # t_out += (P*tS) V + P dV; the K/dK loads overlap these GEMMs.
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv, mma_params.acc_TO, tOrPtS, mma_params.tOrVt,
            smem_copy_params.tOsVt[None, None, None, 0], smem_copy_params.smem_thr_copy_V,
        )
        sm80_utils.gemm_rs(
            mma_params.thr_mma_pv, mma_params.acc_TO, tOrP, mma_params.tOrdVt,
            smem_copy_params.tOsdVt[None, None, None, 0], smem_copy_params.smem_thr_copy_V,
        )
