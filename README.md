# fa4-jax

FlashAttention-4 (Tri Dao's CuTe DSL kernels) as a JAX library, for the
SM80/SM120 family (Ampere → consumer Blackwell) and SM100 (datacenter
Blackwell: B200/GB200, via the tcgen05 kernels), built on
[cutejax](../cutejax).

What it provides over `jax.nn.dot_product_attention`:

- **Native two-sided sliding windows** — `window_size=(left, right)` takes the
  kernel's local path and *skips* out-of-band KV blocks (cuDNN's fused kernel
  only takes a left window).
- **Flex-attention mask mods** — a `MaskMod` is a callable pytree: its class
  and `static()` fields are baked into the kernel as code via cutejax's
  static-pytree-leaf codegen; its array fields (e.g. document ids) stay
  runtime data, so new values reuse the compiled kernel.
- **Both differentiation modes** — reverse mode runs FA4's backward kernels;
  forward mode (`jax.jvp` / `jax.linearize`) runs a fused tangent kernel
  (`fa4_jax/cute/flash_jvp.py`, ours — upstream has no JVP). Built on custom
  primitives with a linear tangent + transpose rule, not `jax.custom_vjp`,
  which would forbid forward mode.
- **Varlen (packed) attention** — `flash_attn_varlen` takes a batch of
  different-length sequences concatenated along the token axis with no
  padding, delimited by int32 `cu_seqlens` offset arrays (the flash-attention
  varlen convention). One compile serves every packing: token counts *and*
  batch size stay symbolic.
- **A pure-JAX reference backend** with identical semantics (including the
  differentiation wiring) — the ground truth the kernels are tested against,
  and `flash_attn(..., backend="reference")` works anywhere.

```python
import jax.numpy as jnp
from fa4_jax import flash_attn, flash_attn_varlen, document_mask

out = flash_attn(q, k, v, causal=True)                       # (b, s, h, d) bf16/fp16
out = flash_attn(q, k, v, window_size=(128, 128))            # two-sided local
out = flash_attn(q, k, v, mask_mod=document_mask(doc_ids))   # flex: block-diag by (b, s) int32 id
out, lse = flash_attn(q, k, v, return_lse=True)

# Packed varlen: q (total_q, h, d), sequence b is q[cu_q[b]:cu_q[b+1]].
cu = jnp.asarray([0, 17, 300, 428], jnp.int32)               # 3 sequences
out = flash_attn_varlen(q, k, v, cu, cu, causal=True)        # (total_q, h, d)
```

Custom masks are frozen dataclasses subclassing `MaskMod`; array fields are
pytree leaves (runtime), anything else is declared `static()`:

```python
from dataclasses import dataclass
from fa4_jax import MaskMod, static

@dataclass(frozen=True)
class PrefixLM(MaskMod):
    prefix_len: jax.Array          # (b,) runtime data
    window: int = static()         # compile-time constant
    def __call__(self, b, h, q, kv):
        return (kv < self.prefix_len[b]) | ((kv <= q) & (q - kv <= self.window))
```

`causal` and `window_size` are bottom-right aligned when `s_q != s_kv` (the
flash-attention convention: the last query lines up with the last key, so
decode / chunked prefill just works). Mask mods see raw indices, so
`mask_mod=causal_mask()` is top-left and only equals `causal=True` when the
lengths match.

Varlen semantics: attention never crosses sequences; `causal`/`window_size`
apply per sequence, bottom-right aligned within it (`cu_seqlens_q` and
`cu_seqlens_k` may pack differently — e.g. one query per sequence against
long caches). Mask mods see the sequence index as `b` and sequence-local
`q_idx`/`kv_idx`. The LSE comes out `(heads, total_q)`. `cu_seqlens` contents
are runtime data and unchecked: they must be nondecreasing, start at 0, and
end at the token count (uncovered tokens get unspecified outputs/gradients).
Zero-length sequences are fine. Reverse mode runs the FA4 varlen backward;
forward mode (`jax.jvp`) is reference-backend only for now (the tangent
kernel has no packed layout); `vmap` over a varlen call is rejected — the
packing *is* the batch.

Restrictions (kernel-side, all checked with clear errors): fp16/bf16, head_dim
a multiple of 8 and ≤ 128 (the SM80/SM120 tile budget; on sm_100, larger heads
would need 2-CTA MMAs, i.e. cluster launches, which this wrapper doesn't do),
no cotangent through `lse` in reverse mode. GQA works in all three passes (the
backward accumulates dK/dV in fp32 across the q heads of each kv head). On
sm_100 the backward runs the tcgen05 kernel for equal head dims and falls back
to the SM80-MMA kernel for mixed ones (which miscompile in upstream b26). Mask
mods work in all three passes — the SM80/SM120 backward's mask_mod support is
a local addition to the vendored kernel (the SM100 one has it upstream).
Combining
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
uv run pytest              # kernels vs the jnp reference (fwd/grad/jvp/masks) + API checks; needs a CUDA device
```

`conftest.py` sets `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4` so the suite shares the GPU.
