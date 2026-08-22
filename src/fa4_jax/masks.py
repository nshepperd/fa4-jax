"""Flex-attention mask mods that run on both backends.

A :class:`MaskMod` wraps one predicate function written against the operations
the two execution contexts share:

- **CuTe codegen** (GPU): the wrapper in ``fa4_jax.kernels`` calls ``fn`` at
  kernel trace time with scalar ``cutlass.Int32`` indices and CuTe tensors for
  aux, so arithmetic (``+ - * // %``), comparisons, and ``& |`` stage into the
  kernel, and ``aux[i][b, q]`` is a scalar gather. The mask is *code*: it rides
  the cutejax pytree as a static leaf, so each distinct MaskMod object compiles
  its own kernel (build them once at module scope or memoize, exactly like
  upstream flex-attention).
- **JAX reference** (any device): ``fn`` is called once with broadcastable
  index arrays (b, h, q, kv axes) and jnp aux arrays; the same operators and
  ``aux[i][b, q]`` advanced indexing produce the dense boolean mask.

So the predicate must stick to: arithmetic/comparison/bitwise ops on the index
args and Python ints, and integer indexing of aux tensors by the index args.
No ``if``/``and``/``or`` on values derived from the indices (Python control
flow would burn the trace); use ``&``/``|`` and arithmetic instead.

The predicate returns True where attention is *allowed*.

Signature::

    fn(b, h, q_idx, kv_idx, aux) -> bool

where ``aux`` is the tuple of aux tensors passed to ``flash_attn`` (their
values are runtime data — one compiled kernel serves any aux contents).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MaskMod", "and_masks", "causal_mask", "document_mask", "or_masks", "sliding_window_mask"]


@dataclass(frozen=True, eq=False)
class MaskMod:
    """A backend-portable mask predicate; see the module docstring.

    Hashed by identity (``eq=False``): the object is a compile-cache key on
    the cute backend, so reuse one instance per distinct mask rather than
    rebuilding it per call.
    """

    fn: Callable[..., Any]
    name: str = field(default="mask_mod")

    def __call__(self, b, h, q_idx, kv_idx, aux):
        return self.fn(b, h, q_idx, kv_idx, aux)

    def __repr__(self):
        return f"MaskMod({self.name}@{id(self):#x})"


def causal_mask() -> MaskMod:
    """``kv_idx <= q_idx``.

    Prefer ``flash_attn(..., causal=True)`` — that takes the kernel's native
    causal path, which also *skips* future KV blocks; this mask-mod variant
    computes and masks them. It exists for composing with other mask mods and
    for exercising the flex path in tests.
    """
    return MaskMod(lambda b, h, q, kv, aux: kv <= q, name="causal")


def sliding_window_mask(left: int | None, right: int | None = 0) -> MaskMod:
    """Two-sided window: allow when ``-right <= q - kv <= left``; None = unbounded.

    Same convention as ``flash_attn``'s native ``window_size`` (which, again,
    skips out-of-band blocks and should be preferred when the window is all
    you need).
    """
    if left is None and right is None:
        raise ValueError("a fully unbounded sliding_window_mask is no mask at all")
    if left is not None and right is not None:
        fn = lambda b, h, q, kv, aux: ((q - kv) <= left) & ((kv - q) <= right)
    elif left is not None:
        fn = lambda b, h, q, kv, aux: (q - kv) <= left
    else:
        fn = lambda b, h, q, kv, aux: (kv - q) <= right
    return MaskMod(fn, name=f"window({left},{right})")


def document_mask(aux_index: int = 0) -> MaskMod:
    """Block-diagonal by document id: ``aux[i][b, q] == aux[i][b, kv]``.

    ``aux[aux_index]`` must be an integer array of shape ``(batch, seqlen)``.
    """
    return MaskMod(
        lambda b, h, q, kv, aux: aux[aux_index][b, q] == aux[aux_index][b, kv],
        name=f"document(aux{aux_index})",
    )


def and_masks(*masks: MaskMod) -> MaskMod:
    """Conjunction of mask mods (attention allowed only where all allow)."""
    if not masks:
        raise ValueError("and_masks needs at least one mask")

    def fn(b, h, q, kv, aux):
        pred = masks[0].fn(b, h, q, kv, aux)
        for m in masks[1:]:
            pred = pred & m.fn(b, h, q, kv, aux)
        return pred

    return MaskMod(fn, name=" & ".join(m.name for m in masks))


def or_masks(*masks: MaskMod) -> MaskMod:
    """Disjunction of mask mods (attention allowed where any allows)."""
    if not masks:
        raise ValueError("or_masks needs at least one mask")

    def fn(b, h, q, kv, aux):
        pred = masks[0].fn(b, h, q, kv, aux)
        for m in masks[1:]:
            pred = pred | m.fn(b, h, q, kv, aux)
        return pred

    return MaskMod(fn, name=" | ".join(m.name for m in masks))
