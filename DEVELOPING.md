# Developing fa4-jax

## Layout

- `src/fa4_jax/attention.py` — public API, custom primitives, jvp/transpose rules.
- `src/fa4_jax/kernels.py` — the cute backend: cutejax plumbing around the kernels.
- `src/fa4_jax/masks.py` — dual-backend `MaskMod` (one predicate, kernel codegen + jnp).
- `src/fa4_jax/reference.py` — executable spec; ground truth for the tests.
- `src/fa4_jax/cute/` — vendored `flash_attn.cute` (see its `VENDOR.md`) plus
  our `flash_jvp.py` tangent kernel; `cute/quack/` is the vendored subset of
  quack's helpers.
- `ci/` — release-time checks for GPUs no local machine has (see `RELEASE.md`).

## Tests

```sh
uv run pytest                    # kernels vs the jnp reference (fwd/grad/jvp/masks) + API checks; needs a CUDA device
uv run pytest --compile-timing   # also report where compile time goes (cutejax builds vs XLA)
```

Kernel-facing tests run the device's own kernels, so covering every
generation means running the suite on each kind of hardware (`RELEASE.md`).

`tests/conftest.py` sets `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.2` so the suite shares the GPU, turns off
XLA's GPU autotuning (it dominated cold-cache runs and an oracle doesn't need
it), and keeps a persistent JAX compile cache in `/tmp/jax_cache_fa4`
(`FA4_JAX_CACHE_DIR` to point it elsewhere, e.g. an empty dir to time a cold
run).

## Which kernel runs where

Each device runs its own family: the SM80-MMA kernels (cp.async + mma.sync)
on sm_8x and sm_12x, wgmma + TMA on sm_90, tcgen05 on sm_100. Tile sizes
come from the device's shared-memory budget: 99 KB parts (sm_86/89, sm_120)
step down to smaller tiles above head dim 128 (64x64 forward; a 64x32 /
64x16 / 32x32 ladder in the backward, `_bwd_cfgs_sm80`), sm_80 keeps 128x64
up to hd 256, sm_90 uses upstream's per-head-dim tables.

The sm_90 and sm_100 backwards fall back to the SM80-MMA backward (the family
runs there too) for shapes their native kernels can't take — see
`_bwd_plan` in `kernels.py`:

- sm_100: mixed head dims (`head_dim != head_dim_v`; tcgen05 miscompiled them
  in flash-attn-4 4.0.0b26 — may be fixed on current cutlass-dsl), and head
  dims whose 16-padded size isn't a multiple of 32 (16, 48, …; the tcgen05
  dQ reduction works in 32-column chunks).
- sm_90: GQA with mixed head dims (the wgmma backward asserts), configs
  whose shared memory doesn't fit, and head dims above 192 that aren't
  multiples of 64 (e.g. 224).

The fallback is correct but slower than the native kernel on that hardware.

## Local kernel changes

Beyond the vendored kernels' own fixes (marked `fa4-jax fix` in the source
and listed in `cute/VENDOR.md`): mask_mod support in the SM80/SM120
backward is a local addition (the SM90 and SM100 ones have it upstream), and
combining causal/window_size with a mask_mod runs the mod as a second masking
pass over the visited tiles (a local `mask.py` change — upstream's branch
structure silently drops the mod in that combination).
