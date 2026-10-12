# fa4-jax

FlashAttention-4 (Tri Dao's CuTe DSL kernels) as a JAX library, for the
SM80/SM120 family (Ampere → consumer Blackwell), SM90 (Hopper: H100/H200,
via the wgmma + TMA kernels) and SM100 (datacenter Blackwell: B200/GB200,
via the tcgen05 kernels), built on [cutejax](https://github.com/nshepperd/cutejax).

```sh
pip install "fa4-jax[cu13]"   # jax[cuda13] + the CuTe DSL's CUDA 13 runtime libraries
```

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
Zero-length sequences are fine. Optional `seqused_q`/`seqused_k` (int32
`(batch,)`) limit each slot to its first `seqused[b]` tokens — the KV-cache
case, where slots are allocated larger than currently filled; unused K is
invisible, unused Q rows get unspecified outputs/gradients, and the
band alignment uses the used lengths. Reverse mode runs the FA4 varlen
backward and forward mode the fused varlen tangent kernel; `vmap` over a
varlen call is rejected — the packing *is* the batch.

## Supported hardware and limits

- **GPUs**: Ampere and Ada (sm_80/86/89), Hopper (sm_90), datacenter Blackwell
  (sm_100) and consumer Blackwell (sm_120). Each GPU runs the kernel family
  written for it.
- **dtypes**: fp16 and bf16.
- **Head dims**: multiples of 8, up to 256 — except on datacenter Blackwell
  (sm_100) and in forward mode (`jax.jvp`), where the limit is 128. `head_dim`
  and `head_dim_v` may differ. Anything unsupported raises a clear
  `ValueError` up front rather than failing inside CUDA.
- **GQA** works in the forward, backward and forward-mode passes.
- **Mask mods** work in all three passes and combine with
  `causal`/`window_size`, which keep their block skipping. The mod itself
  can't skip blocks yet (flex block sparsity isn't wired up), so a very
  sparse mask still visits every tile.
- **Not supported**: differentiating through the returned `lse` in reverse
  mode (`return_lse=True` outputs are fine; forward mode handles them).

Working on fa4-jax itself: see `DEVELOPING.md` and `RELEASE.md`.

## License

fa4-jax's own code is MIT (`LICENSE`). The vendored kernels keep their
upstream licenses: flash-attention's CuTe DSL kernels in `src/fa4_jax/cute/`
are BSD-3-Clause (`LICENSE.flash-attention`), and quack's helpers in
`src/fa4_jax/cute/quack/` are Apache-2.0 (`LICENSE.quack`).
