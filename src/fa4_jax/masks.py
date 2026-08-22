"""Flex-attention mask mods that run on both backends.

A mask mod is a **callable pytree**: a :class:`MaskMod` subclass whose array
fields are the runtime data the predicate indexes (document ids, segment
lengths, ...) and whose remaining fields are compile-time constants. Calling
it with ``(b, h, q_idx, kv_idx)`` returns True where attention is *allowed*.

``flash_attn`` flattens the mask: the array leaves become ordinary operands
of the attention primitive (traced, vmappable, runtime data), and the treedef
— the class plus its static fields — becomes the static kernel-cache key. So
two ``document_mask(ids_a)`` / ``document_mask(ids_b)`` built on different
calls share one compiled kernel, and the predicate's *code* is pinned by the
class rather than by object identity.

The predicate body is written once against the operations the two execution
contexts share:

- **CuTe codegen** (GPU): the wrapper in ``fa4_jax.kernels`` rebuilds the mask
  from its treedef with CuTe tensors as the leaves and calls it at kernel
  trace time with scalar ``cutlass.Int32`` indices, so arithmetic
  (``+ - * // %``), comparisons, and ``& |`` stage into the kernel, and
  ``self.ids[b, q]`` is a scalar gather.
- **JAX reference** (any device): the mask is called once with broadcastable
  index arrays (b, h, q, kv axes) and jnp leaves; the same operators and
  ``self.ids[b, q]`` advanced indexing produce the dense boolean mask.

So the predicate must stick to: arithmetic/comparison/bitwise ops on the index
args and static Python ints, and integer indexing of array fields by the index
args. No ``if``/``and``/``or`` on values derived from the indices (Python
control flow would burn the trace); use ``&``/``|`` and arithmetic instead.

Defining a mask::

    @dataclass(frozen=True)
    class DocumentMask(MaskMod):
        ids: Array                      # array field -> pytree leaf, runtime
        def __call__(self, b, h, q, kv):
            return self.ids[b, q] == self.ids[b, kv]

    @dataclass(frozen=True)
    class Window(MaskMod):
        left: int = static()            # static field -> in the treedef
        def __call__(self, b, h, q, kv):
            return q - kv <= self.left

Non-static fields must hold arrays (``flash_attn`` rejects Python scalars in
leaf position — mark them ``static()``). Subclasses are registered as pytrees
automatically; unflattening bypasses ``__init__``, so constructors must not
validate their fields (the GPU path rebuilds them with CuTe tensors).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax

__all__ = [
    "FnMask",
    "MaskMod",
    "and_masks",
    "causal_mask",
    "document_mask",
    "or_masks",
    "sliding_window_mask",
    "static",
]

_STATIC = "fa4_static"


def static(**kwargs) -> Any:
    """A dataclass field that lives in the treedef (compile-time constant)."""
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_STATIC] = True
    return field(metadata=metadata, **kwargs)


class MaskMod:
    """Base class for mask mods; see the module docstring.

    Subclass as a ``@dataclass(frozen=True)`` and implement ``__call__``.
    Array fields are pytree children; fields declared with :func:`static`
    ride in the treedef and must be hashable.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)

    def __call__(self, b, h, q_idx, kv_idx):
        raise NotImplementedError

    @classmethod
    def _field_split(cls):
        dyn, stat = [], []
        for f in dataclasses.fields(cls):
            (stat if f.metadata.get(_STATIC) else dyn).append(f.name)
        return tuple(dyn), tuple(stat)

    def _tree_flatten(self):
        dyn, stat = self._field_split()
        return (
            tuple(getattr(self, n) for n in dyn),
            tuple(getattr(self, n) for n in stat),
        )

    @classmethod
    def _tree_unflatten(cls, aux, children):
        dyn, stat = cls._field_split()
        obj = object.__new__(cls)
        for n, v in zip(dyn, children):
            object.__setattr__(obj, n, v)
        for n, v in zip(stat, aux):
            object.__setattr__(obj, n, v)
        return obj

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self):
        return f"{self.name}({', '.join(f'{f.name}={getattr(self, f.name)!r}' for f in dataclasses.fields(self))})"


@dataclass(frozen=True)
class FnMask(MaskMod):
    """Wrap a bare predicate ``fn(b, h, q_idx, kv_idx) -> bool`` with no data.

    ``fn`` is static, hashed by identity: build the FnMask once (module
    scope or memoized) so the cute backend keys one kernel per predicate.
    """

    fn: Callable[..., Any] = static()  # noqa: RUF009
    label: str = static(default="fn")

    def __call__(self, b, h, q_idx, kv_idx):
        return self.fn(b, h, q_idx, kv_idx)

    @property
    def name(self) -> str:
        return self.label


@dataclass(frozen=True)
class CausalMask(MaskMod):
    """``kv_idx <= q_idx``.

    Prefer ``flash_attn(..., causal=True)`` — that takes the kernel's native
    causal path, which also *skips* future KV blocks; this mask-mod variant
    computes and masks them. It exists for composing with other mask mods and
    for exercising the flex path in tests.
    """

    def __call__(self, b, h, q_idx, kv_idx):
        return kv_idx <= q_idx


@dataclass(frozen=True)
class SlidingWindowMask(MaskMod):
    """Two-sided window: allow when ``-right <= q - kv <= left``; None = unbounded.

    Same convention as ``flash_attn``'s native ``window_size`` (which, again,
    skips out-of-band blocks and should be preferred when the window is all
    you need).
    """

    left: int | None = static()
    right: int | None = static()

    def __call__(self, b, h, q_idx, kv_idx):
        if self.left is not None and self.right is not None:
            return ((q_idx - kv_idx) <= self.left) & ((kv_idx - q_idx) <= self.right)
        if self.left is not None:
            return (q_idx - kv_idx) <= self.left
        return (kv_idx - q_idx) <= self.right


@dataclass(frozen=True)
class DocumentMask(MaskMod):
    """Block-diagonal by document id: ``ids[b, q] == ids[b, kv]``.

    ``ids`` is an integer array of shape ``(batch, seqlen)``.
    """

    ids: Any

    def __call__(self, b, h, q_idx, kv_idx):
        return self.ids[b, q_idx] == self.ids[b, kv_idx]


@dataclass(frozen=True)
class AndMask(MaskMod):
    masks: tuple[MaskMod, ...]

    def __call__(self, b, h, q_idx, kv_idx):
        pred = self.masks[0](b, h, q_idx, kv_idx)
        for m in self.masks[1:]:
            pred = pred & m(b, h, q_idx, kv_idx)
        return pred


@dataclass(frozen=True)
class OrMask(MaskMod):
    masks: tuple[MaskMod, ...]

    def __call__(self, b, h, q_idx, kv_idx):
        pred = self.masks[0](b, h, q_idx, kv_idx)
        for m in self.masks[1:]:
            pred = pred | m(b, h, q_idx, kv_idx)
        return pred


def causal_mask() -> MaskMod:
    """See :class:`CausalMask`."""
    return CausalMask()


def sliding_window_mask(left: int | None, right: int | None = 0) -> MaskMod:
    """See :class:`SlidingWindowMask`."""
    if left is None and right is None:
        raise ValueError("a fully unbounded sliding_window_mask is no mask at all")
    return SlidingWindowMask(left, right)


def document_mask(ids) -> MaskMod:
    """See :class:`DocumentMask`; ``ids`` is the ``(batch, seqlen)`` int array."""
    return DocumentMask(ids)


def and_masks(*masks: MaskMod) -> MaskMod:
    """Conjunction of mask mods (attention allowed only where all allow)."""
    if not masks:
        raise ValueError("and_masks needs at least one mask")
    return AndMask(tuple(masks))


def or_masks(*masks: MaskMod) -> MaskMod:
    """Disjunction of mask mods (attention allowed where any allows)."""
    if not masks:
        raise ValueError("or_masks needs at least one mask")
    return OrMask(tuple(masks))
