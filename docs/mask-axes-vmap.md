# Design: axis specs on MaskMod leaves, for vmap folding

Status: proposal (2026-08-23), not implemented. Follows the callable-pytree
`MaskMod` refactor (array fields = primitive operands, treedef = kernel key).

## Problem

`flash_attn` batches under `vmap` by *folding*: the mapped axis is moved next
to the batch axis and merged, `(vmap, B, ...) -> (vmap*B, ...)`, then the same
primitive is re-bound. That keeps the custom JVP/transpose rules in the graph
and maps straight onto the kernels' batch parallelism.

Folding is only done when `cfg.mask is None`. With a mask we fall back to
`jax.vmap` over the backend impl, which (a) differentiates through the
backend's jnp code rather than the primitives — so on the cute backend it is
unsupported — and (b) is conservative even when it *would* be fine: the
common document-mask case has `ids: (B, S)`, whose leading axis is exactly the
batch axis q/k/v fold into.

Two things stop the fold from being automatic:

1. **We don't know which axis of a leaf is the batch axis** (or that it has
   none — a `(S,)` array shared across the batch).
2. **The predicate may use `b` other than to index a leaf.** Folding renumbers
   `b` over `vmap*B`; `ids[b, q]` stays consistent because `ids` was folded the
   same way, but `b % 2 == 0` changes meaning.

## Proposal

### Leaves declare their axes

An array field declares, per dimension, which attention axis it corresponds
to — or `None` for an unrelated axis:

```python
@dataclass(frozen=True)
class DocumentMask(MaskMod):
    ids: Array = array("B", "S")          # (batch, seqlen); S == Q == K

@dataclass(frozen=True)
class PrefixMask(MaskMod):
    prefix_len: Array = array("B")        # (batch,)
    stride: int = static()

@dataclass(frozen=True)
class AlibiLike(MaskMod):
    per_head_bias: Array = array("H", None)   # (heads, anything) — no batch axis
```

`array(*axes)` is a field helper like `static()`: it stores the axes tuple in
the field metadata. Vocabulary:

| label | meaning | checked against |
|---|---|---|
| `"B"` | batch | `q.shape[0]` |
| `"H"` | query heads | `q.shape[2]` |
| `"Q"` | query positions | `q.shape[1]` |
| `"K"` | key positions | `k.shape[1]` |
| `"S"` | positions indexed by both `q_idx` and `kv_idx` | requires `s_q == s_kv`, equals both |
| `None` | unrelated | nothing |

Declaring axes is also a **promise** about the predicate: `b` and `h` are
used only to index leaves along their `"B"`/`"H"` axes (never in arithmetic
or comparisons), so renumbering them under a fold is harmless.

### Foldability is derived

A mask is *foldable* iff every array field of every `MaskMod` node in it has
declared axes. So:

- `CausalMask`, `SlidingWindowMask` (no array fields): foldable vacuously.
  Correct — they don't look at `b`.
- `FnMask` (wraps arbitrary code): sets `foldable = False` explicitly; it
  can't make the promise.
- `AndMask`/`OrMask`: foldable iff all children are (recursion through the
  `masks` field, which holds sub-masks rather than arrays).

`flash_attn` flattens the mask with key paths
(`jax.tree_util.tree_flatten_with_path`; `MaskMod` registers with
`register_pytree_with_keys` so the path's last key is the field name) and
computes one entry per leaf:

```python
AttnConfig.aux_batch_axes: tuple[int | None, ...] | None
#   None            -> mask not foldable (today's behaviour)
#   (0, None, ...)  -> per leaf: position of its "B" axis, or None if it has none
```

That's all the batching rule needs; the full axes tuple is consumed at the
API boundary for validation.

### Batching rule

```python
def _fold_leaf(a, d, p, size):
    """a: leaf (with vmap axis at d, or d is None); p: its batch-axis position."""
    if d is None:
        if p is None:
            return a                                   # shared across batch; stays shared
        a = jnp.broadcast_to(a[None], (size, *a.shape))
        a = jnp.moveaxis(a, 0, p)                      # vmap axis just before B
    else:
        if p is None:
            raise _Unfoldable                          # mapped data with no batch axis to merge into
        a = jnp.moveaxis(a, d, p)                      # B now sits at p+1 whatever d was
    return a.reshape(*a.shape[:p], size * a.shape[p + 1], *a.shape[p + 2:])
```

`_fwd_batch`/`_tangent_batch` fold q/k/v (and the tangent's extra operands)
exactly as now, fold the aux leaves with `_fold_leaf` using
`cfg.aux_batch_axes`, and rebind. The vmap axis is merged *major* in both
cases, so `b` indexes consistently. If `aux_batch_axes is None`, or a mapped
leaf has no `"B"` axis, fall back to `_batch_via_impl` as today (reference
works; cute raises its existing unsupported error — but now with a message
naming the leaf and saying "declare a B axis or don't vmap it").

The case "mapped leaf without a B axis" is genuinely unfoldable: the
predicate indexes it *without* `b`, so after vmap each instance wants
different data at the same `(q, kv)` and only `b` could select it. The user
fix is to give it a batch axis (`array("B", "S")` with a broadcasted array),
which costs nothing in `jnp` and makes the fold legal.

### Shape validation (the other payoff)

The cute kernels wrap `q_idx`/`kv_idx` modulo the sequence lengths before
any aux gather (`compute_fastdiv_mods`), so a wrongly-sized leaf doesn't
crash — it silently produces a wrong mask. With axes declared, `flash_attn`
can check every labelled dimension against q/k at trace time and raise:

```
DocumentMask.ids: axis 1 labelled "S" has size 512 but seqlen is 256
```

This alone justifies the labels beyond `"B"`.

### Heads (extension, not in scope)

`"H"` lets vmap over a head axis fold into axis 2 of q/k/v by the same rule.
Heads are a static dim in the cutejax specs, so that fold recompiles per
vmapped head count; correct but not free. Leave the rule at `"B"` first.

## What changes where

- `masks.py`: `array(*axes)` helper; `MaskMod` registers with keys; a
  `foldable` classvar (default derived, `FnMask` opts out); built-ins
  annotated (`DocumentMask.ids = array("B", "S")`).
- `attention.py`: `flash_attn` computes `aux_batch_axes` + validates shapes;
  `AttnConfig` gains the field; `_batch_via_fold` takes per-operand batch
  positions; `_fwd_batch`/`_tangent_batch` choose fold when possible.
- `kernels.py`: nothing (leaves arrive folded like q/k/v).
- tests: `jax.vmap(flash_attn, in_axes=(0, 0, 0, DocumentMask(0)))` on cute
  equals a Python loop; unmapped mask under vmap; shared `(S,)` leaf unmapped
  folds, mapped raises with the helpful message; shape-mismatch error.

## Alternatives considered

- **A method `batch_axis(self) -> pytree`** instead of field metadata:
  equivalent information, but metadata sits beside the field and needs no
  per-class code. Keep the method as an override hook for rank-dependent
  cases if one shows up.
- **Infer the batch axis from shape** (`shape[0] == B`): ambiguous whenever
  `B == S` or `B == H`, and silently wrong exactly in the small test cases
  where everyone first tries it. Rejected.
- **Always fold, no declaration**: breaks any predicate using `b`
  arithmetically, with no diagnostic. Rejected — the promise must be explicit.
