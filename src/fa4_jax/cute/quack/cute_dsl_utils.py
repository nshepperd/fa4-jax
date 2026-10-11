# Copyright (c) 2025, Tri Dao.
# Vendored from quack-kernels 0.6.4 (Dao-AILab/quack, Apache-2.0; see
# LICENSE.quack at the repo root). Modified by fa4-jax: reduced to the parts
# the vendored kernels use -- the TVM-FFI Constexpr converter patch and
# ParamsBase -- which drops upstream's module-level `import torch` (and the
# torch dtype maps, device queries, cubin/compile hooks and NamedTuple
# protocol that needed it or that nothing here calls).

from dataclasses import dataclass, fields
from typing import get_origin

import cutlass
from cutlass.base_dsl.tvm_ffi_builder import spec
from cutlass.cutlass_dsl import NumericMeta

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


# Patch TVM-FFI converter to handle Constexpr type annotations as compile-time constants.
# Fields annotated with cutlass.Constexpr[T] are emitted as ConstNone (not runtime args).
# At call time, pass None for these fields; the compile-time value is baked in.
import cutlass.cute._tvm_ffi_args_spec_converter as _converter_module  # noqa: E402

_original_convert_single_arg = _converter_module._convert_single_arg


def _patched_convert_single_arg(arg, arg_name, arg_type, ctx):
    if arg_type is not None and get_origin(arg_type) is cutlass.Constexpr:
        return spec.ConstNone(arg_name)
    # If arg is a NamedTuple but arg_type doesn't have _fields (e.g. annotated as tuple),
    # redirect so the converter uses the NamedTuple's own type hints.
    if (
        isinstance(arg, tuple)
        and hasattr(type(arg), "_fields")
        and (arg_type is None or not hasattr(arg_type, "_fields"))
    ):
        return _original_convert_single_arg(arg, arg_name, type(arg), ctx)
    return _original_convert_single_arg(arg, arg_name, arg_type, ctx)


_converter_module._convert_single_arg = _patched_convert_single_arg


def _partition_fields(obj):
    """Split dataclass fields into (constexpr_dict, non_constexpr_dict) by type."""
    all_fields = {field.name: getattr(obj, field.name) for field in fields(obj)}
    constexpr = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
    non_constexpr = {n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)}
    return constexpr, non_constexpr


def _new_from_mlir_values(self, values):
    constexpr_fields, non_constexpr_fields = _partition_fields(self)
    for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
        non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
        values = values[n_items:]
    return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        _, non_constexpr_fields = _partition_fields(self)
        values, self._values_pos = [], []
        for obj in non_constexpr_fields.values():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    __new_from_mlir_values__ = _new_from_mlir_values
