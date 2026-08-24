# Vendored flash_attn.cute

Source: the `flash-attn-4` wheel, version 4.0.0b26 (Dao-AILab/flash-attention,
BSD-3-Clause — see `LICENSE.flash-attention` at the repo root), specifically
the copy installed in minimax-music3's venv, which already carried the
`fa4-local-window-block-skip.patch` speed fix (SM80/SM120 forward: skip KV
blocks left of the sliding window instead of masking them; upstream marks the
gap with a bare `# TODO: local`).

The SM80/SM120 (Ampere / consumer-Blackwell) and SM100 (datacenter
Blackwell: `flash_fwd_sm100.py`, `flash_bwd_sm100.py` and the helpers they
drag in — `blackwell_helpers.py`, `mma_sm100_desc.py`, `barrier.py`,
`copy_utils.py`, `fa_logging.py`) forward+backward closures are vendored —
not the SM90 kernels, the SM100 MLA/hd256-2CTA variants, the torch
`interface.py`, or the varlen/paged serving extras beyond what the closures
drag in. The SM100 files were taken from the same 4.0.0b26 wheel
(re-downloaded from PyPI) with only the import rewrite applied.

## Local changes

- All `flash_attn.cute` imports rewritten to `fa4_jax.cute`.
- `cute_dsl_utils.py`, `block_sparsity.py`: `import torch` made optional
  (guarded); only their torch host-side helpers need it, which fa4-jax never
  calls. `from __future__ import annotations` added where torch types appear
  in annotations.
- `flash_fwd.py`: the lazy `FlashAttentionForwardSm90` re-export removed
  (module not vendored).
- `flash_jvp.py`: **new file, not upstream** — forward-mode (JVP) tangent
  kernel, written for fa4-jax in the style of `flash_bwd.py`'s recompute
  pass. See its docstring for the math. SM80-style (cp.async + mma.sync),
  but those instructions exist on every supported arch, so the same kernel
  also serves sm_100/sm_120.
- `flash_bwd.py` (SM80/SM120 backward): mask_mod + aux tensors added
  (upstream's SM100 backward already has them). Also a varlen+GQA fix: the
  dK/dV atomic-accumulate path offset its fp32 accumulators by the *raw*
  `offset_k + batch_idx * n_block_size` while the postprocess kernel (and
  the dQaccum path in the same kernel) use `SeqlenInfoQK`'s tile-rounded
  padded offset — correct only when every sequence length is a tile
  multiple. Now uses `seqlen.padded_offset_k`.
- `mask.py`: the FlexAttention mask_mod element loops extracted into methods
  (`apply_mask_mod_sm80`, `apply_mask_mod_sm100_transposed`), and every
  `apply_mask*` entry point runs the mod as a second masking pass when it is
  combined with the native causal/local band — upstream's branch structure
  would silently drop the mod in that combination. The band's KV-block
  skipping is unaffected: ANDing a mod onto the band can only mask more.

## Updating

Diff against `flash_attn/cute/` of a newer flash-attn-4 wheel, re-apply the
changes above (git log this directory for their exact shape), and re-run the
GPU test suite.
