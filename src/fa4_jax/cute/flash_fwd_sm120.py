# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# SM120 uses the same SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has
# a smaller shared memory capacity (99 KB vs 163 KB on SM80). This module subclasses
# FlashAttentionForwardSm80 and overrides the SMEM capacity check accordingly.

import cutlass.utils as utils_basic
from cutlass.base_dsl.arch import Arch

from fa4_jax.cute.flash_fwd import FlashAttentionForwardSm80


class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    def __init__(self, *args, **kwargs):
        """Force SM80 code paths while the DSL still targets the resident SM120 GPU."""
        super().__init__(*args, **kwargs)
        self.arch = Arch.sm_80

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
        smem_capacity=None,
    ) -> bool:
        """Same checks as SM80; the capacity defaults to SM120's 99 KB."""
        if smem_capacity is None:
            smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        return FlashAttentionForwardSm80.can_implement(
            dtype, head_dim, head_dim_v, tile_m, tile_n, num_stages, num_threads,
            is_causal, Q_in_regs, smem_capacity=smem_capacity,
        )
