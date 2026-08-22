"""FlashAttention-4 (CuTeDSL) for JAX.

Public surface:

- ``flash_attn`` — attention with causal/sliding-window/flex masking,
  reverse-mode (FA4 backward kernels) *and* forward-mode (fused JVP tangent
  kernel) differentiation.
- ``MaskMod`` and the mask combinators in ``fa4_jax.masks`` — flex-attention
  style mask mods that run both as CuTe codegen (GPU) and as plain JAX (the
  reference backend).
- ``reference_attn`` — the pure-JAX reference implementation the kernels are
  tested against; also usable on CPU as a drop-in.

Importing ``fa4_jax`` itself stays lightweight: the CuTe kernels (cutlass,
cutejax) load lazily on first GPU-backend call, so the reference backend and
the unit tests work on machines with no GPU stack at all.
"""

from fa4_jax.attention import flash_attn
from fa4_jax.masks import MaskMod, causal_mask, document_mask, sliding_window_mask
from fa4_jax.reference import reference_attn

__all__ = [
    "MaskMod",
    "causal_mask",
    "document_mask",
    "flash_attn",
    "reference_attn",
    "sliding_window_mask",
]
