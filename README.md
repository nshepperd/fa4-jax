# fa4-jax

FlashAttention-4 (Tri Dao's CuTe DSL kernels) as a JAX library, for the
SM80/SM120 family (Ampere → consumer Blackwell), built on
[cutejax](../cutejax).

What it provides over `jax.nn.dot_product_attention`:

- **Native two-sided sliding windows** — `window_size=(left, right)` takes the
  kernel's local path and *skips* out-of-band KV blocks (cuDNN's fused kernel
  only takes a left window).
- **Flex-attention mask mods** — a `MaskMod` predicate is baked into the
  kernel as code via cutejax's static-pytree-leaf codegen; aux tensors (e.g.
  document ids) stay runtime data, so new values reuse the compiled kernel.
- **Both differentiation modes** — reverse mode runs FA4's backward kernels;
  forward mode (`jax.jvp` / `jax.linearize`) runs a fused tangent kernel
  (`fa4_jax/cute/flash_jvp.py`, ours — upstream has no JVP). Built on custom
  primitives with a linear tangent + transpose rule, not `jax.custom_vjp`,
  which would forbid forward mode.
- **A pure-JAX reference backend** with identical semantics (including the
  differentiation wiring), so everything above the kernels is unit-tested on
  CPU, and `flash_attn(..., backend="reference")` works anywhere.

```python
import jax.numpy as jnp
from fa4_jax import flash_attn, document_mask

out = flash_attn(q, k, v, causal=True)                       # (b, s, h, d) bf16/fp16
out = flash_attn(q, k, v, window_size=(128, 128))            # two-sided local
out = flash_attn(q, k, v, mask_mod=document_mask(),          # flex: block-diag by id
                 aux_tensors=(doc_ids,))                     # (b, s) int32
out, lse = flash_attn(q, k, v, return_lse=True)
```

Restrictions (kernel-side, all checked with clear errors): fp16/bf16, head_dim
a multiple of 8, GQA forward-and-JVP-only (repeat kv heads for reverse-mode
gradients). Mask mods work in all three passes — the backward's mask_mod
support is a local addition to the vendored kernel. Combining
causal/window_size with a mask_mod keeps the native band's KV-block skipping
and runs the mod as a second masking pass over the visited tiles (also a
local mask.py change; upstream's branch structure silently drops the mod in
that combination). The mod itself can't skip blocks — that would need flex
block sparsity, which is not wired up.

## Layout

- `src/fa4_jax/attention.py` — public API, custom primitives, jvp/transpose rules.
- `src/fa4_jax/kernels.py` — the cute backend: cutejax plumbing around the kernels.
- `src/fa4_jax/masks.py` — dual-backend `MaskMod` (one predicate, kernel codegen + jnp).
- `src/fa4_jax/reference.py` — executable spec; ground truth for the tests.
- `src/fa4_jax/cute/` — vendored `flash_attn.cute` (see its `VENDOR.md`) plus
  our `flash_jvp.py` tangent kernel.

## Tests

```sh
uv run pytest              # CPU: reference vs jax.nn.dot_product_attention, wiring, hypothesis shapes
uv run pytest -m gpu       # kernels vs reference, on a CUDA device
```
