# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) backward pass.
#
# SM120 uses the same SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has
# a smaller shared memory capacity (99 KB vs 163 KB on SM80). This module subclasses
# FlashAttentionBackwardSm80 and overrides the SMEM capacity check accordingly.

import cutlass.utils as utils_basic

from fa4_jax.cute.flash_bwd import FlashAttentionBackwardSm80


class FlashAttentionBackwardSm120(FlashAttentionBackwardSm80):
    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        n_block_size,
        num_stages_Q,
        num_stages_dO,
        num_threads,
        is_causal,
        V_in_regs=False,
        smem_capacity=None,
    ) -> bool:
        """Same checks as SM80; the capacity defaults to SM120's 99 KB."""
        if smem_capacity is None:
            smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        return FlashAttentionBackwardSm80.can_implement(
            dtype, head_dim, head_dim_v, m_block_size, n_block_size,
            num_stages_Q, num_stages_dO, num_threads, is_causal, V_in_regs,
            smem_capacity=smem_capacity,
        )
