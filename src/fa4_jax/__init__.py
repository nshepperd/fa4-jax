"""FlashAttention-4 (CuTeDSL) for JAX.

Public surface:

- ``flash_attn`` — attention with causal/sliding-window/flex masking,
  reverse-mode (FA4 backward kernels) *and* forward-mode (fused JVP tangent
  kernel) differentiation.
- ``flash_attn_varlen`` — the same over packed variable-length batches:
  ``(total_tokens, heads, head_dim)`` tensors delimited by int32
  ``cu_seqlens`` offset arrays (plus optional ``seqused`` used-length
  overrides for partially-filled slots), no padding. Both differentiation
  modes run on the kernels, like ``flash_attn``.
- ``MaskMod`` (a callable-pytree base class: array fields are runtime data,
  ``static()`` fields are compile-time) and the masks/combinators in
  ``fa4_jax.masks`` — flex-attention style mask mods that run both as CuTe
  codegen (GPU) and as plain JAX (the reference backend).
- ``reference_attn`` — the pure-JAX reference implementation the kernels are
  tested against; also usable on CPU as a drop-in.

Importing ``fa4_jax`` itself stays lightweight: the CuTe kernels (cutlass,
cutejax) load lazily on first GPU-backend call, so the reference backend and
the unit tests work on machines with no GPU stack at all.
"""

from fa4_jax.attention import default_kernel_arch, flash_attn, flash_attn_varlen
from fa4_jax.masks import (
    FnMask,
    MaskMod,
    and_masks,
    causal_mask,
    document_mask,
    or_masks,
    sliding_window_mask,
    static,
)
from fa4_jax.reference import reference_attn, reference_attn_varlen

__all__ = [
    "FnMask",
    "MaskMod",
    "and_masks",
    "causal_mask",
    "default_kernel_arch",
    "document_mask",
    "flash_attn",
    "flash_attn_varlen",
    "or_masks",
    "reference_attn",
    "reference_attn_varlen",
    "sliding_window_mask",
    "static",
]
