# Vendored flash_attn.cute

Source: the `flash-attn-4` wheel, version 4.0.0b26 (Dao-AILab/flash-attention,
BSD-3-Clause — see `LICENSE.flash-attention` at the repo root), specifically
the copy installed in minimax-music3's venv, which already carried the
`fa4-local-window-block-skip.patch` speed fix (SM80/SM120 forward: skip KV
blocks left of the sliding window instead of masking them; upstream marks the
gap with a bare `# TODO: local`).

Only the SM80/SM120 (Ampere / consumer-Blackwell) forward+backward closure is
vendored — not the SM90/SM100 kernels, the torch `interface.py`, or the
varlen/paged serving extras beyond what the closure drags in.

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
  pass. See its docstring for the math.

## Updating

Diff against `flash_attn/cute/` of a newer flash-attn-4 wheel, re-apply the
changes above (git log this directory for their exact shape), and re-run the
GPU test suite.
