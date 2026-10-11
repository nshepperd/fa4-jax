# Vendored flash_attn.cute

Source: the `flash-attn-4` wheel, version 4.0.0b26 (Dao-AILab/flash-attention,
BSD-3-Clause — see `LICENSE.flash-attention` at the repo root), specifically
the copy installed in minimax-music3's venv, which already carried the
`fa4-local-window-block-skip.patch` speed fix (SM80/SM120 forward: skip KV
blocks left of the sliding window instead of masking them; upstream marks the
gap with a bare `# TODO: local`).

The SM80/SM120 (Ampere / consumer-Blackwell), SM90 (Hopper:
`flash_fwd_sm90.py`, `flash_bwd_sm90.py`; their helpers — `pipeline.py`,
`named_barrier.py`, `barrier.py`, `pack_gqa.py`, `paged_kv.py`,
`block_sparse_utils.py` — were already here, and the wgmma-level helpers
come from `quack.sm90_utils`) and SM100 (datacenter Blackwell:
`flash_fwd_sm100.py`, `flash_bwd_sm100.py` and the helpers they drag in —
`blackwell_helpers.py`, `mma_sm100_desc.py`, `barrier.py`, `copy_utils.py`,
`fa_logging.py`) forward+backward closures are vendored — not the SM100
MLA/hd256-2CTA variants, the torch `interface.py`, or the varlen/paged
serving extras beyond what the closures drag in. The SM100 and SM90 files
were taken from the same 4.0.0b26 wheel (re-downloaded from PyPI; the SM90
kernels are byte-identical in the latest release, 4.0.0b29) with the import
rewrite applied, then the local changes below. `interface.py`'s per-head-dim
SM90 tile tables (`_tile_size_fwd_sm90` / `_tile_size_bwd_sm90`) are
reproduced in `fa4_jax/kernels.py`.

## Local changes

- All `flash_attn.cute` imports rewritten to `fa4_jax.cute`.
- `cute_dsl_utils.py`, `block_sparsity.py`: `import torch` made optional
  (guarded); only their torch host-side helpers need it, which fa4-jax never
  calls. `from __future__ import annotations` added where torch types appear
  in annotations.
- `flash_fwd.py`: the lazy `FlashAttentionForwardSm90` re-export removed
  (module not vendored). Also the kernel body is guarded on
  `work_tile.is_valid_tile` (as upstream's backward kernels already are):
  the varlen scheduler's grid is an upper bound and its wasted tiles decode
  to `batch_idx == num_batch`, for which `SeqlenInfoQK` reads
  `cu_seqlens[num_batch + 1]` / `seqused[num_batch]` one element out of
  bounds and then acts on garbage seqlens — observed as illegal-address
  faults from the derived JVP kernel; upstream's comment claims those tiles
  see zero seqlens, but nothing enforces it. The epilogue's smem O tile is
  staged at the *base* of the shared-memory struct instead of at `sQ`
  (upstream): sO is `(tile_m, head_dim_v)` while sQ holds
  `(tile_m, head_dim)`, so `head_dim_v > head_dim` overflowed into sK and —
  beyond ~1.5x — past the allocation entirely (smem OOB,
  `CUDA_ERROR_ILLEGAL_ADDRESS`); `can_implement` now also bounds the O tile
  by the struct size. And `use_tma_O` is pinned False:
  upstream enables it for sm_90..<sm_120 *devices*, but this kernel never
  builds a TMA atom for O (the epilogue gets `tma_atom_O=None`) — the flag
  only made sense for the unvendored Sm90 subclass, and it misfires whenever
  this family runs on an sm_100 device (kernel_arch override, or by analogy
  with the backward's mixed-head-dims fallback).
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

- `flash_fwd_sm100.py`, `flash_bwd_sm100.py`: varlen TMA loads clamped to
  ``seqused`` via ragged descriptors (upstream loads the full boundary tile,
  so NaN/Inf in the dead padding of a partially-used slot poisons used rows:
  V through the PV MMA's ``0*NaN`` in the forward, Q/K/V through the bwd
  MMAs — score masking cannot contain the ``dK = dS^T Q`` / ``dQ = dS K``
  operand paths). K/V (fwd) and Q/K/V/dO (bwd) are wrapped with
  ``copy_utils.create_ragged_tensor_for_tma`` + per-sequence
  ``offset_ragged_tensor`` so rows past ``seqused`` are descriptor-OOB and
  zero-fill in hardware. The local ``copy_utils`` versions are used, not
  quack's: quack's builds the wraparound extents/strides as static ints,
  which overflows the DSL's static layout arithmetic
  (``std::bad_variant_access``) whenever the head dims are compile-time —
  ours materializes them as runtime values. The fwd prologue also clamps its
  first K/V load to block 0 (upstream loads block ``n_block_max - 1 = -1``
  when ``seqused_k == 0``, reading the previous slot's tail). The 2-CTA
  Qt/Kt/dOt paths (unused here) are left raw. The SM80/SM120 family needs
  none of this: its cp.async loads already fold the row predicate in, and a
  false predicate zero-fills.

- `flash_fwd_sm90.py` (Hopper forward): the same three varlen/OOB fixes
  the other families needed, in this kernel's shape. (1) Varlen K/V TMA
  loads are ragged (our `copy_utils.create_ragged_tensor_for_tma` +
  per-sequence `offset_ragged_tensor` in `load`) so rows past `seqused_k`
  zero-fill instead of loading a partially-used slot's dead padding (NaN
  there reaches the PV MMA as `0*NaN`; upstream only rags the O *store*,
  through quack's `ptr_shift` flavor, which is kept). (2) The first K/V
  load is clamped to block 0: upstream lets the TMA path load block
  `n_block_max - 1 = -1` when `seqused_k == 0` (or a local band lies
  entirely left of the keys) on the theory that TMA zero-fills it, but a
  varlen sequence is a domain offset into the packed tensor, so block -1 is
  the previous slot's tail. (3) The epilogue's O tile is staged over sV
  when it does not fit in sQ (upstream always uses sQ, so `head_dim_v >
  head_dim` overflowed into sK and beyond); sV/sQ/sK are contiguous and
  every tile is dead by then. Plus a real `can_implement`: the inherited
  one is the SM80 kernel's (SM80 tile formula against sm_80's 163 KB), so
  it mirrors `_get_shared_storage_cls` against sm_90's 227 KB and bounds
  the O tile — upstream's largest configs sit exactly at the limit, and a
  launch that overflows poisons the CUDA context.
- `flash_bwd_sm90.py` (Hopper backward): (1) head dims are padded to a
  multiple of 32 (upstream: 16). `tile_hdim` is also the fp32 accumulator
  geometry (dQaccum blocks are `tile_m * tile_hdim`, GQA's dK/dV blocks
  `tile_n * tile_hdim`), and the pre-/postprocess kernels that zero and
  convert those buffers pad to 32: with 16, head dims that are 16 mod 32
  (48, 80, 112, ...) had the kernel and the postprocess disagree on every
  block's offset and on the fragment layout, i.e. garbage dQ (and dK/dV
  under GQA). Verified on an H100: as shipped, head_dim 80 gives dQ with a
  max error of 1.2 against the reference (dK/dV, which take the direct TMA
  store path, are fine); with 32 it matches to bf16 tolerance. Upstream's SM100
  backward pads to 16 too and shares the postprocess, so the same head dims
  are suspect there (untested here: no sm_100 device). (2) Varlen Q/K/V/dO TMA loads are ragged like the
  forward's (upstream rags only the dK/dV stores) — `dK = dS^T Q` and
  `dQ = dS K` multiply the raw operands, so score masking cannot contain
  NaN padding. (3) `can_implement` replaced: upstream's was never called by
  its interface and rejects every real config (`2 * tile_m % num_threads`
  with 384 threads) while checking no SMEM budget; ours takes the
  constructor's arguments and mirrors `_get_shared_storage_cls` field by
  field (`smem_usage_bytes`). (4) The dO smem layout's atom width follows
  `head_dim_v` (upstream: `head_dim`, for both Q and dO) -- with
  `head_dim_v < head_dim`, e.g. (64, 32), the 64-wide atom cannot tile the
  dO tile and TMA-atom creation fails at compile time.

- `flash_bwd_preprocess.py`: the head-dim-OOB predicate for the O/dO loads
  is sliced per row (`tOpO[None, m, None]`, as `flash_fwd.load_Q` does)
  instead of passing the whole `(rest_v, CPY_M, CPY_K)` tensor to each
  row's copy. The unsliced form fails IR verification with a predicate
  shape mismatch whenever the padded head dim gives one copy chunk per row
  (`CPY_K == 1`), i.e. head dims 16 and 48 could not compile the backward
  on any family; 80 (three chunks) happened to pass.

## Updating

Diff against `flash_attn/cute/` of a newer flash-attn-4 wheel, re-apply the
changes above (git log this directory for their exact shape), and re-run the
GPU test suite.

## quack (`quack/`)

The kernels' `quack` imports (`copy_utils`, `layout_utils`, `sm90_utils`,
`utils`, and `ParamsBase` from `cute_dsl_utils`) now point at a vendored
subset of quack-kernels 0.6.4 (Dao-AILab/quack, Apache-2.0 — see
`LICENSE.quack` at the repo root) in `fa4_jax/cute/quack/`, with imports
rewritten. Vendored rather than depended on because quack-kernels pins
`nvidia-cutlass-dsl==4.6.2` (cutejax needs >= 4.8, so a PyPI install couldn't
resolve; this repo only got away with it via a uv override) and requires
torch, which `quack.cute_dsl_utils` also imported at module level. Changes,
also noted in each file's header: `cute_dsl_utils.py` is cut down to the
TVM-FFI `Constexpr` converter patch (kept: importing it was a side effect the
kernels may rely on) and `ParamsBase`, dropping torch; `copy_utils.py` drops
`sr_cvt_copy` (stochastic rounding, needs `quack.rounding`, unused here).
