# Copyright 2026 Shumpei Shiina.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import egrpc
import jax
import jax.numpy as jnp
import numpy as np

from ..alignment import (
    assert_alignment_signature,
    assert_normalized_distance,
    assert_privacy_axis,
    reshape_alignment_signature,
)
from ..array_base import SensitiveDimInt
from ..util import DPError
from .array import PrivArray, SensitiveArray
from .domain import ArrayDomain
from .util import assert_same_distance_and_accountant


PrimitiveValue = PrivArray | SensitiveArray | jax.Array
ShapeDimension = int | str | SensitiveDimInt


def _concrete_dimension(dimension: ShapeDimension) -> int:
    if isinstance(dimension, str):
        raise ValueError(
            "Symbolic dimensions must be bound before primitive execution."
        )
    if isinstance(dimension, SensitiveDimInt):
        return int(dimension._value)
    return dimension


def _concrete_shape(shape: tuple[ShapeDimension, ...]) -> tuple[int, ...]:
    return tuple(_concrete_dimension(dimension) for dimension in shape)


def _scale_value_range(
    value_range: tuple[float | None, float | None] | None,
    scalar: float,
) -> tuple[float | None, float | None] | None:
    if value_range is None:
        return None
    lo, hi = value_range
    if scalar >= 0:
        return (
            lo * scalar if lo is not None else None,
            hi * scalar if hi is not None else None,
        )
    return (
        hi * scalar if hi is not None else None,
        lo * scalar if lo is not None else None,
    )


def _shift_value_range(
    value_range: tuple[float | None, float | None] | None,
    scalar: float | None,
) -> tuple[float | None, float | None] | None:
    if value_range is None or scalar is None:
        return None
    lo, hi = value_range
    return (
        lo + scalar if lo is not None else None,
        hi + scalar if hi is not None else None,
    )


def _subtract_value_range(
    value_range: tuple[float | None, float | None] | None,
    scalar: float | None,
) -> tuple[float | None, float | None] | None:
    if value_range is None or scalar is None:
        return None
    lo, hi = value_range
    return (
        lo - scalar if lo is not None else None,
        hi - scalar if hi is not None else None,
    )


def _reverse_subtract_value_range(
    scalar: float | None,
    value_range: tuple[float | None, float | None] | None,
) -> tuple[float | None, float | None] | None:
    if value_range is None or scalar is None:
        return None
    lo, hi = value_range
    return (
        scalar - hi if hi is not None else None,
        scalar - lo if lo is not None else None,
    )


def _concrete_scalar(value: jax.Array) -> float | None:
    if isinstance(value, jax.core.Tracer):
        return None
    array = np.asarray(jax.device_get(value))
    if array.ndim != 0:
        return None
    return float(array)


def _preserved_private_axis(
    x: PrivArray,
    result: jax.Array,
    primitive: str,
) -> int:
    rank_offset = result.ndim - x._value.ndim
    if rank_offset < 0:
        raise DPError(
            f"{primitive} cannot remove private operand dimensions."
        )
    return rank_offset + x._privacy_axis


def _assert_public_broadcast(
    private: PrivArray,
    public: jax.Array,
    primitive: str,
) -> None:
    output_ndim = (
        private._value.ndim
        if private._value.ndim >= public.ndim
        else public.ndim
    )
    output_privacy_axis = (
        output_ndim - private._value.ndim + private._privacy_axis
    )
    public_axis = output_privacy_axis - (output_ndim - public.ndim)
    if public_axis >= 0 and public.shape[public_axis] != 1:
        raise DPError(
            f"{primitive} cannot match a public dimension against the "
            "private record count."
        )


def _require_none(name: str, value: Any) -> None:
    if value is not None:
        raise NotImplementedError(f"{name} is not supported.")


def _mul_value(
    x: jax.Array,
    y: jax.Array,
    out_dtype: str | None,
) -> jax.Array:
    _require_none("mul out_dtype", out_dtype)
    return jax.lax.mul(x, y)


@egrpc.multifunction
def add(x: PrivArray, y: jax.Array) -> PrivArray:
    _assert_public_broadcast(x, y, "add")
    return _binary_private_public(
        x,
        jax.lax.add(x._value, y),
        "add",
        domain=ArrayDomain(
            value_range=_shift_value_range(
                x._domain.value_range,
                _concrete_scalar(y),
            )
        ),
    )


@add.register
def _(x: jax.Array, y: PrivArray) -> PrivArray:
    _assert_public_broadcast(y, x, "add")
    return _binary_private_public(
        y,
        jax.lax.add(x, y._value),
        "add",
        domain=ArrayDomain(
            value_range=_shift_value_range(
                y._domain.value_range,
                _concrete_scalar(x),
            )
        ),
    )


@add.register
def _(x: PrivArray, y: PrivArray) -> PrivArray:
    return _binary_private_private(
        x,
        y,
        jax.lax.add(x._value, y._value),
        "add",
    )


@add.register(remote=False)
def _(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.add(x, y)


@egrpc.multifunction
def add_any(x: PrivArray, y: jax.Array) -> PrivArray:
    _assert_public_broadcast(x, y, "add_any")
    return _binary_private_public(
        x,
        jax.lax.add(x._value, y),
        "add_any",
        domain=ArrayDomain(
            value_range=_shift_value_range(
                x._domain.value_range,
                _concrete_scalar(y),
            )
        ),
    )


@add_any.register
def _(x: jax.Array, y: PrivArray) -> PrivArray:
    _assert_public_broadcast(y, x, "add_any")
    return _binary_private_public(
        y,
        jax.lax.add(x, y._value),
        "add_any",
        domain=ArrayDomain(
            value_range=_shift_value_range(
                y._domain.value_range,
                _concrete_scalar(x),
            )
        ),
    )


@add_any.register
def _(x: PrivArray, y: PrivArray) -> PrivArray:
    return _binary_private_private(
        x,
        y,
        jax.lax.add(x._value, y._value),
        "add_any",
    )


@add_any.register(remote=False)
def _(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.add(x, y)


@egrpc.multifunction
def sub(x: PrivArray, y: jax.Array) -> PrivArray:
    _assert_public_broadcast(x, y, "sub")
    return _binary_private_public(
        x,
        jax.lax.sub(x._value, y),
        "sub",
        domain=ArrayDomain(
            value_range=_subtract_value_range(
                x._domain.value_range,
                _concrete_scalar(y),
            )
        ),
    )


@sub.register
def _(x: jax.Array, y: PrivArray) -> PrivArray:
    _assert_public_broadcast(y, x, "sub")
    return _binary_private_public(
        y,
        jax.lax.sub(x, y._value),
        "sub",
        domain=ArrayDomain(
            value_range=_reverse_subtract_value_range(
                _concrete_scalar(x),
                y._domain.value_range,
            )
        ),
    )


@sub.register
def _(x: PrivArray, y: PrivArray) -> PrivArray:
    return _binary_private_private(
        x,
        y,
        jax.lax.sub(x._value, y._value),
        "sub",
    )


@sub.register(remote=False)
def _(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.sub(x, y)


@egrpc.multifunction
def mul(
    x: PrivArray,
    y: jax.Array,
    out_dtype: str | None = None,
) -> PrivArray:
    _assert_public_broadcast(x, y, "mul")
    scalar = _concrete_scalar(y)
    value_range = (
        _scale_value_range(x._domain.value_range, scalar)
        if scalar is not None
        else None
    )
    return _binary_private_public(
        x,
        _mul_value(x._value, y, out_dtype),
        "mul",
        domain=ArrayDomain(value_range=value_range),
    )


@mul.register
def _(
    x: jax.Array,
    y: PrivArray,
    out_dtype: str | None = None,
) -> PrivArray:
    _assert_public_broadcast(y, x, "mul")
    scalar = _concrete_scalar(x)
    value_range = (
        _scale_value_range(y._domain.value_range, scalar)
        if scalar is not None
        else None
    )
    return _binary_private_public(
        y,
        _mul_value(x, y._value, out_dtype),
        "mul",
        domain=ArrayDomain(value_range=value_range),
    )


@mul.register
def _(
    x: PrivArray,
    y: PrivArray,
    out_dtype: str | None = None,
) -> PrivArray:
    return _binary_private_private(
        x,
        y,
        _mul_value(x._value, y._value, out_dtype),
        "mul",
    )


@mul.register(remote=False)
def _(
    x: jax.Array,
    y: jax.Array,
    out_dtype: str | None = None,
) -> jax.Array:
    return _mul_value(x, y, out_dtype)


@egrpc.multifunction
def broadcast_in_dim(
    x: PrivArray,
    shape: tuple[ShapeDimension, ...],
    broadcast_dimensions: tuple[int, ...],
    sharding: str | None = None,
) -> PrivArray:
    _require_none("broadcast_in_dim sharding", sharding)
    concrete_shape = _concrete_shape(shape)
    if len(broadcast_dimensions) != x._value.ndim:
        raise ValueError(
            "broadcast_dimensions must contain one entry per input dimension."
        )
    if (
        tuple(sorted(broadcast_dimensions)) != broadcast_dimensions
        or len(set(broadcast_dimensions)) != len(broadcast_dimensions)
        or any(
            not 0 <= axis < len(concrete_shape)
            for axis in broadcast_dimensions
        )
    ):
        raise ValueError(
            "broadcast_dimensions must be unique, sorted output dimensions."
        )

    output_privacy_axis = broadcast_dimensions[x._privacy_axis]
    private_dimensions = [
        (axis, dimension)
        for axis, dimension in enumerate(shape)
        if isinstance(dimension, SensitiveDimInt)
    ]
    if (
        len(private_dimensions) != 1
        or private_dimensions[0][0] != output_privacy_axis
        or private_dimensions[0][1]._scale != 1
        or private_dimensions[0][1]._alignment_signature
        != x._alignment_signature
    ):
        raise DPError(
            "broadcast_in_dim may only preserve the aligned privacy "
            "dimension."
        )

    return PrivArray(
        value=jax.lax.broadcast_in_dim(
            x._value,
            concrete_shape,
            broadcast_dimensions,
        ),
        distance=x._distance,
        privacy_axis=output_privacy_axis,
        domain=ArrayDomain(value_range=x._domain.value_range),
        parents=[x],
        keep_alignment=True,
    )


@broadcast_in_dim.register
def _(
    x: jax.Array,
    shape: tuple[ShapeDimension, ...],
    broadcast_dimensions: tuple[int, ...],
    sharding: str | None = None,
) -> PrivArray:
    _require_none("broadcast_in_dim sharding", sharding)
    private_dimensions = [
        (axis, dimension)
        for axis, dimension in enumerate(shape)
        if isinstance(dimension, SensitiveDimInt)
    ]
    if len(private_dimensions) != 1:
        raise DPError(
            "broadcast_in_dim requires one private-shaped dimension."
        )
    privacy_axis = private_dimensions[0][0]
    if privacy_axis in broadcast_dimensions:
        public_axis = broadcast_dimensions.index(privacy_axis)
        if public_axis < x.ndim and x.shape[public_axis] != 1:
            raise DPError(
                "broadcast_in_dim cannot match a public dimension "
                "against the private record count."
            )

    concrete_shape = _concrete_shape(shape)
    value = jax.lax.broadcast_in_dim(
        x,
        concrete_shape,
        broadcast_dimensions,
    )
    privacy_axis, dimension = private_dimensions[0]
    if dimension._scale <= 0:
        raise DPError("A private-shaped dimension must have a positive scale.")
    result = PrivArray(
        value=value,
        distance=dimension._distance,
        privacy_axis=privacy_axis,
        domain=ArrayDomain(),
        accountant=dimension._accountant,
    )
    if dimension._scale == 1:
        result._alignment_signature = dimension._alignment_signature
    return result


@broadcast_in_dim.register(remote=False)
def _(
    x: jax.Array,
    shape: tuple[int, ...],
    broadcast_dimensions: tuple[int, ...],
    sharding: str | None = None,
) -> jax.Array:
    _require_none("broadcast_in_dim sharding", sharding)
    return jax.lax.broadcast_in_dim(
        x,
        shape,
        broadcast_dimensions,
    )


DimensionNumbers = tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
]


def _dot_output_privacy_axis(
    *,
    side: str,
    privacy_axis: int,
    lhs_ndim: int,
    rhs_ndim: int,
    dimensions: DimensionNumbers,
) -> int:
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimensions
    contracting = lhs_contract if side == "lhs" else rhs_contract
    batch = lhs_batch if side == "lhs" else rhs_batch
    if privacy_axis in contracting:
        raise DPError("dot_general cannot contract the privacy dimension.")
    if privacy_axis in batch:
        return batch.index(privacy_axis)

    lhs_remaining = [
        axis
        for axis in range(lhs_ndim)
        if axis not in lhs_contract and axis not in lhs_batch
    ]
    rhs_remaining = [
        axis
        for axis in range(rhs_ndim)
        if axis not in rhs_contract and axis not in rhs_batch
    ]
    if side == "lhs":
        return len(lhs_batch) + lhs_remaining.index(privacy_axis)
    return len(lhs_batch) + len(lhs_remaining) + rhs_remaining.index(privacy_axis)


def _dot_value(
    lhs: jax.Array,
    rhs: jax.Array,
    dimensions: DimensionNumbers,
    preferred_element_type: str | None,
) -> jax.Array:
    return jax.lax.dot_general(
        lhs,
        rhs,
        dimensions,
        precision=None,
        preferred_element_type=(
            None if preferred_element_type is None else np.dtype(preferred_element_type)
        ),
    )


def _dot_private_public(
    lhs: PrivArray,
    rhs: jax.Array,
    dimensions: DimensionNumbers,
    preferred_element_type: str | None,
) -> PrivArray:
    if lhs._privacy_axis in dimensions[1][0]:
        raise DPError(
            "dot_general cannot pair a private dimension with a public "
            "batch dimension."
        )
    privacy_axis = _dot_output_privacy_axis(
        side="lhs",
        privacy_axis=lhs._privacy_axis,
        lhs_ndim=lhs._value.ndim,
        rhs_ndim=rhs.ndim,
        dimensions=dimensions,
    )
    return PrivArray(
        value=_dot_value(lhs._value, rhs, dimensions, preferred_element_type),
        distance=lhs._distance,
        privacy_axis=privacy_axis,
        domain=ArrayDomain(),
        parents=[lhs],
        keep_alignment=True,
    )


def _dot_public_private(
    lhs: jax.Array,
    rhs: PrivArray,
    dimensions: DimensionNumbers,
    preferred_element_type: str | None,
) -> PrivArray:
    if rhs._privacy_axis in dimensions[1][1]:
        raise DPError(
            "dot_general cannot pair a public batch dimension with a "
            "private dimension."
        )
    privacy_axis = _dot_output_privacy_axis(
        side="rhs",
        privacy_axis=rhs._privacy_axis,
        lhs_ndim=lhs.ndim,
        rhs_ndim=rhs._value.ndim,
        dimensions=dimensions,
    )
    return PrivArray(
        value=_dot_value(lhs, rhs._value, dimensions, preferred_element_type),
        distance=rhs._distance,
        privacy_axis=privacy_axis,
        domain=ArrayDomain(),
        parents=[rhs],
        keep_alignment=True,
    )


def _dot_private_private(
    lhs: PrivArray,
    rhs: PrivArray,
    dimensions: DimensionNumbers,
    preferred_element_type: str | None,
) -> PrivArray | SensitiveArray:
    assert_alignment_signature(lhs, rhs)
    assert_same_distance_and_accountant(lhs, rhs)
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimensions
    if (
        lhs._privacy_axis in lhs_batch
        and rhs._privacy_axis in rhs_batch
        and lhs_batch.index(lhs._privacy_axis)
        == rhs_batch.index(rhs._privacy_axis)
    ):
        privacy_axis = lhs_batch.index(lhs._privacy_axis)
        return PrivArray(
            value=_dot_value(
                lhs._value,
                rhs._value,
                dimensions,
                preferred_element_type,
            ),
            distance=lhs._distance,
            privacy_axis=privacy_axis,
            domain=ArrayDomain(),
            parents=[lhs, rhs],
            keep_alignment=True,
        )

    if (
        lhs._privacy_axis in lhs_contract
        and rhs._privacy_axis in rhs_contract
        and lhs_contract.index(lhs._privacy_axis)
        == rhs_contract.index(rhs._privacy_axis)
    ):
        lhs_bound = lhs._domain.norm_bound
        rhs_bound = rhs._domain.norm_bound
        if lhs_bound is None:
            raise DPError("Left dot_general operand has no per-record norm bound.")
        if rhs_bound is None:
            raise DPError("Right dot_general operand has no per-record norm bound.")
        return SensitiveArray(
            value=_dot_value(
                lhs._value,
                rhs._value,
                dimensions,
                preferred_element_type,
            ),
            distance=lhs._distance * lhs_bound * rhs_bound,
            norm_type="l2",
            parents=[lhs, rhs],
        )

    raise DPError(
        "dot_general between private operands must pair their aligned "
        "privacy dimensions as matching batch or contracting dimensions."
    )


@egrpc.multifunction
def dot_general(
    lhs: PrivArray,
    rhs: jax.Array,
    dimension_numbers: DimensionNumbers,
    precision: str | None,
    preferred_element_type: str | None,
    out_sharding: str | None,
) -> PrivArray:
    _require_none("dot_general precision", precision)
    _require_none("dot_general out_sharding", out_sharding)
    return _dot_private_public(
        lhs,
        rhs,
        dimension_numbers,
        preferred_element_type,
    )


@dot_general.register
def _(
    lhs: jax.Array,
    rhs: PrivArray,
    dimension_numbers: DimensionNumbers,
    precision: str | None,
    preferred_element_type: str | None,
    out_sharding: str | None,
) -> PrivArray:
    _require_none("dot_general precision", precision)
    _require_none("dot_general out_sharding", out_sharding)
    return _dot_public_private(
        lhs,
        rhs,
        dimension_numbers,
        preferred_element_type,
    )


@dot_general.register
def _(
    lhs: PrivArray,
    rhs: PrivArray,
    dimension_numbers: DimensionNumbers,
    precision: str | None,
    preferred_element_type: str | None,
    out_sharding: str | None,
) -> PrivArray | SensitiveArray:
    _require_none("dot_general precision", precision)
    _require_none("dot_general out_sharding", out_sharding)
    return _dot_private_private(
        lhs,
        rhs,
        dimension_numbers,
        preferred_element_type,
    )


@dot_general.register(remote=False)
def _(
    lhs: jax.Array,
    rhs: jax.Array,
    dimension_numbers: DimensionNumbers,
    precision: str | None,
    preferred_element_type: str | None,
    out_sharding: str | None,
) -> jax.Array:
    _require_none("dot_general precision", precision)
    _require_none("dot_general out_sharding", out_sharding)
    return _dot_value(
        lhs,
        rhs,
        dimension_numbers,
        preferred_element_type,
    )


ConvDimensionNumbers = jax.lax.ConvDimensionNumbers
# egrpc wire form; the trusted endpoint reconstructs JAX's named tuple.
ConvDimensionNumbersParam = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]
ConvPadding = tuple[tuple[int, int], ...]
SymbolicGroupCount = int | str | SensitiveDimInt


def _conv_dimension_numbers(
    dimension_numbers: ConvDimensionNumbersParam,
) -> ConvDimensionNumbers:
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    rank = len(lhs_spec)
    if (
        rank < 3
        or sorted(lhs_spec) != list(range(rank))
        or sorted(rhs_spec) != list(range(rank))
        or sorted(out_spec) != list(range(rank))
    ):
        raise ValueError(
            "Convolution dimension specifications must be equal-rank "
            "permutations."
        )
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def _conv_output_shape(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    window_strides: tuple[int, ...],
    padding: ConvPadding,
    lhs_dilation: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    dimensions: ConvDimensionNumbers,
) -> tuple[int, ...]:
    spatial_rank = len(window_strides)
    rank = spatial_rank + 2
    if (
        len(lhs_shape) != rank
        or len(rhs_shape) != rank
        or len(padding) != spatial_rank
        or len(lhs_dilation) != spatial_rank
        or len(rhs_dilation) != spatial_rank
    ):
        raise ValueError(
            "Convolution operands and spatial parameters have inconsistent "
            "ranks."
        )
    if any(
        value <= 0
        for value in (
            *window_strides,
            *lhs_dilation,
            *rhs_dilation,
        )
    ):
        raise ValueError(
            "Convolution strides and dilations must be positive."
        )

    lhs_canonical = tuple(lhs_shape[axis] for axis in dimensions.lhs_spec)
    rhs_canonical = tuple(rhs_shape[axis] for axis in dimensions.rhs_spec)
    spatial_shape: list[int] = []
    for lhs_size, rhs_size, stride, pads, lhs_rate, rhs_rate in zip(
        lhs_canonical[2:],
        rhs_canonical[2:],
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        strict=True,
    ):
        lhs_effective = (lhs_size - 1) * lhs_rate + 1
        rhs_effective = (rhs_size - 1) * rhs_rate + 1
        size = (
            lhs_effective + pads[0] + pads[1] - rhs_effective
        ) // stride + 1
        if size < 0:
            raise ValueError("Convolution has a negative output dimension.")
        spatial_shape.append(size)

    canonical = (
        lhs_canonical[0],
        rhs_canonical[0],
        *spatial_shape,
    )
    output = [0] * rank
    for canonical_axis, physical_axis in enumerate(dimensions.out_spec):
        output[physical_axis] = canonical[canonical_axis]
    return tuple(output)


def _conv_value(
    lhs: jax.Array,
    rhs: jax.Array,
    window_strides: tuple[int, ...],
    padding: ConvPadding,
    lhs_dilation: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    dimensions: ConvDimensionNumbers,
    feature_group_count: int,
    batch_group_count: int,
    preferred_element_type: str | None,
) -> jax.Array:
    output_shape = _conv_output_shape(
        tuple(int(dim) for dim in lhs.shape),
        tuple(int(dim) for dim in rhs.shape),
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimensions,
    )
    if feature_group_count == 0:
        dtype = (
            jnp.result_type(lhs, rhs)
            if preferred_element_type is None
            else jnp.dtype(preferred_element_type)
        )
        return jnp.zeros(output_shape, dtype=dtype)
    return jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimensions,
        feature_group_count=feature_group_count,
        batch_group_count=batch_group_count,
        precision=None,
        preferred_element_type=(
            None
            if preferred_element_type is None
            else np.dtype(preferred_element_type)
        ),
    )


def _conv_private_public(
    lhs: PrivArray,
    rhs: jax.Array,
    window_strides: tuple[int, ...],
    padding: ConvPadding,
    lhs_dilation: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    dimensions: ConvDimensionNumbers,
    feature_group_count: int,
    batch_group_count: int,
    preferred_element_type: str | None,
) -> PrivArray:
    if lhs._privacy_axis != dimensions.lhs_spec[0]:
        raise DPError(
            "A private convolution lhs must use its batch dimension as the "
            "privacy dimension."
        )
    if batch_group_count != 1:
        raise DPError(
            "Private convolution does not support batch_group_count != 1."
        )
    if feature_group_count <= 0:
        raise DPError(
            "A forward private convolution requires a positive "
            "feature_group_count."
        )
    value = _conv_value(
        lhs._value,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        dimensions,
        feature_group_count,
        batch_group_count,
        preferred_element_type,
    )
    output_privacy_axis = dimensions.out_spec[0]
    return PrivArray(
        value=value,
        distance=lhs._distance,
        privacy_axis=output_privacy_axis,
        domain=ArrayDomain(),
        parents=[lhs],
        keep_alignment=True,
    )


def _concrete_group_count(value: SymbolicGroupCount) -> int:
    if not isinstance(value, int):
        raise DPError(
            "Convolution group counts cannot depend on a private dimension."
        )
    return value


@egrpc.multifunction
def conv_general_dilated(
    lhs: PrivArray,
    rhs: jax.Array,
    window_strides: tuple[int, ...],
    padding: ConvPadding,
    lhs_dilation: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    dimension_numbers: ConvDimensionNumbersParam,
    feature_group_count: SymbolicGroupCount,
    batch_group_count: SymbolicGroupCount,
    precision: str | None,
    preferred_element_type: str | None,
    out_sharding: str | None,
) -> PrivArray:
    _require_none("conv_general_dilated precision", precision)
    _require_none("conv_general_dilated out_sharding", out_sharding)
    if (
        not isinstance(feature_group_count, int)
        or not isinstance(batch_group_count, int)
    ):
        raise DPError(
            "A private-public convolution cannot use a private dimension "
            "as a group count."
        )
    return _conv_private_public(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        _conv_dimension_numbers(dimension_numbers),
        _concrete_group_count(feature_group_count),
        _concrete_group_count(batch_group_count),
        preferred_element_type,
    )


@conv_general_dilated.register
def _(
    lhs: PrivArray,
    rhs: PrivArray,
    window_strides: tuple[int, ...],
    padding: ConvPadding,
    lhs_dilation: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    dimension_numbers: ConvDimensionNumbersParam,
    feature_group_count: SymbolicGroupCount,
    batch_group_count: SymbolicGroupCount,
    precision: str | None,
    preferred_element_type: str | None,
    out_sharding: str | None,
) -> PrivArray:
    _require_none("conv_general_dilated precision", precision)
    _require_none("conv_general_dilated out_sharding", out_sharding)
    dimensions = _conv_dimension_numbers(dimension_numbers)
    if (
        not isinstance(feature_group_count, SensitiveDimInt)
        or not isinstance(batch_group_count, int)
        or batch_group_count != 1
        or lhs._privacy_axis != dimensions.lhs_spec[1]
        or rhs._privacy_axis != dimensions.rhs_spec[0]
        or feature_group_count._alignment_signature
        != lhs._alignment_signature
        or feature_group_count._scale != 1
        or not feature_group_count._distance.structurally_equal(
            lhs._distance
        )
        or lhs._alignment_signature.base
        != rhs._alignment_signature.base
        or lhs._alignment_signature.left
        != rhs._alignment_signature.left
        or (
            rhs._alignment_signature.right
            % lhs._alignment_signature.right
        ) != 0
        or int(rhs._value.shape[dimensions.rhs_spec[1]]) != 1
    ):
        raise DPError(
            "Private-private convolution is supported only for a "
            "record-aligned grouped kernel gradient."
        )
    assert_normalized_distance(
        lhs._distance,
        lhs._alignment_signature,
        rhs._distance,
        rhs._alignment_signature,
    )
    result = PrivArray(
        value=_conv_value(
            lhs._value,
            rhs._value,
            window_strides,
            padding,
            lhs_dilation,
            rhs_dilation,
            dimensions,
            int(feature_group_count._value),
            batch_group_count,
            preferred_element_type,
        ),
        distance=rhs._distance,
        privacy_axis=dimensions.out_spec[1],
        domain=ArrayDomain(),
        parents=[lhs, rhs],
        keep_alignment=False,
    )
    result._alignment_signature = rhs._alignment_signature
    return result


@conv_general_dilated.register(remote=False)
def _(
    lhs: jax.Array,
    rhs: jax.Array,
    window_strides: tuple[int, ...],
    padding: ConvPadding,
    lhs_dilation: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    dimension_numbers: ConvDimensionNumbersParam,
    feature_group_count: SymbolicGroupCount,
    batch_group_count: SymbolicGroupCount,
    precision: str | None,
    preferred_element_type: str | None,
    out_sharding: str | None,
) -> jax.Array:
    _require_none("conv_general_dilated precision", precision)
    _require_none("conv_general_dilated out_sharding", out_sharding)
    if (
        not isinstance(feature_group_count, int)
        or not isinstance(batch_group_count, int)
    ):
        raise DPError(
            "Public convolution group counts cannot depend on a private "
            "dimension."
        )
    concrete_feature_groups = _concrete_group_count(feature_group_count)
    concrete_batch_groups = _concrete_group_count(batch_group_count)
    if concrete_feature_groups < 0 or concrete_batch_groups <= 0:
        raise ValueError(
            "Convolution group counts must be non-negative/positive."
        )
    return _conv_value(
        lhs,
        rhs,
        window_strides,
        padding,
        lhs_dilation,
        rhs_dilation,
        _conv_dimension_numbers(dimension_numbers),
        concrete_feature_groups,
        concrete_batch_groups,
        preferred_element_type,
    )


@egrpc.multifunction
def reshape(
    x: PrivArray,
    new_sizes: tuple[ShapeDimension, ...],
    dimensions: tuple[int, ...] | None,
    sharding: str | None = None,
) -> PrivArray:
    _require_none("reshape sharding", sharding)
    private_dimensions = [
        (axis, dimension)
        for axis, dimension in enumerate(new_sizes)
        if isinstance(dimension, SensitiveDimInt)
    ]
    if private_dimensions:
        if len(private_dimensions) != 1:
            raise DPError(
                "A private reshape must have exactly one privacy dimension."
            )
        output_privacy_axis, output_dimension = private_dimensions[0]
        scale = output_dimension._scale
        if scale <= 0:
            raise ValueError("A private reshape scale must be positive.")
        concrete_sizes = _concrete_shape(new_sizes)
    else:
        if any(isinstance(size, str) for size in new_sizes):
            raise ValueError(
                "Symbolic dimensions must be bound before primitive execution."
            )
        concrete_sizes = tuple(
            int(size)
            for size in new_sizes
            if isinstance(size, int)
        )
        if len(concrete_sizes) != len(new_sizes):
            raise TypeError("Unsupported reshape dimension.")
        if (
            x._privacy_axis >= len(concrete_sizes)
            or concrete_sizes[x._privacy_axis] != -1
        ):
            raise DPError(
                "An eager private reshape must infer its privacy dimension "
                "with -1."
            )
        output_privacy_axis = x._privacy_axis
        scale = 1
        output_dimension = SensitiveDimInt(
            value=int(x._value.shape[x._privacy_axis]),
            distance=x._distance,
            alignment_signature=x._alignment_signature,
            parents=[x],
        )
        mutable_sizes = list(concrete_sizes)
        mutable_sizes[output_privacy_axis] = int(
            x._value.shape[x._privacy_axis]
        )
        concrete_sizes = tuple(mutable_sizes)

    input_shape = tuple(int(dim) for dim in x._value.shape)
    if dimensions is None:
        permuted_shape = input_shape
        input_privacy_axis = x._privacy_axis
    else:
        if sorted(dimensions) != list(range(len(input_shape))):
            raise ValueError(
                "Reshape dimensions must contain each input axis exactly once."
            )
        permuted_shape = tuple(input_shape[axis] for axis in dimensions)
        input_privacy_axis = dimensions.index(x._privacy_axis)
    if any(dimension < 0 for dimension in concrete_sizes):
        raise ValueError("Private reshape dimensions must be non-negative.")

    input_prefix = int(
        np.prod(permuted_shape[:input_privacy_axis], dtype=np.int64)
    )
    input_suffix = int(
        np.prod(permuted_shape[input_privacy_axis + 1 :], dtype=np.int64)
    )
    output_prefix = int(
        np.prod(concrete_sizes[:output_privacy_axis], dtype=np.int64)
    )
    output_suffix = int(
        np.prod(concrete_sizes[output_privacy_axis + 1 :], dtype=np.int64)
    )
    output_alignment = reshape_alignment_signature(
        x._alignment_signature,
        output_dimension._alignment_signature,
        scale,
        input_prefix=input_prefix,
        input_suffix=input_suffix,
        output_prefix=output_prefix,
        output_suffix=output_suffix,
    )
    assert_normalized_distance(
        x._distance,
        x._alignment_signature,
        output_dimension._distance,
        output_alignment,
    )
    preserve_norm_bound = (
        output_alignment.left >= x._alignment_signature.left
        and output_alignment.right >= x._alignment_signature.right
    )
    result = PrivArray(
        value=jax.lax.reshape(x._value, concrete_sizes, dimensions),
        distance=output_dimension._distance,
        privacy_axis=output_privacy_axis,
        domain=ArrayDomain(
            norm_type=x._domain.norm_type,
            norm_bound=(
                x._domain.norm_bound
                if preserve_norm_bound
                else None
            ),
            value_range=x._domain.value_range,
        ),
        parents=[x],
        keep_alignment=(output_alignment == x._alignment_signature),
    )
    result._alignment_signature = output_alignment
    return result


@reshape.register(remote=False)
def _(
    x: jax.Array,
    new_sizes: tuple[ShapeDimension, ...],
    dimensions: tuple[int, ...] | None,
    sharding: str | None = None,
) -> jax.Array:
    _require_none("reshape sharding", sharding)
    if any(
        isinstance(size, (str, SensitiveDimInt))
        for size in new_sizes
    ):
        raise DPError(
            "A public reshape cannot depend on a private dimension."
        )
    return jax.lax.reshape(
        x,
        _concrete_shape(new_sizes),
        dimensions,
    )


@egrpc.multifunction
def transpose(x: PrivArray, permutation: tuple[int, ...]) -> PrivArray:
    if sorted(permutation) != list(range(x._value.ndim)):
        raise ValueError("Transpose permutation must contain each axis exactly once.")
    return PrivArray(
        value=jax.lax.transpose(x._value, permutation),
        distance=x._distance,
        privacy_axis=permutation.index(x._privacy_axis),
        domain=x._domain,
        parents=[x],
        keep_alignment=True,
    )


@transpose.register(remote=False)
def _(x: jax.Array, permutation: tuple[int, ...]) -> jax.Array:
    return jax.lax.transpose(x, permutation)


@egrpc.multifunction
def rev(x: PrivArray, dimensions: tuple[int, ...]) -> PrivArray:
    if (
        len(set(dimensions)) != len(dimensions)
        or any(not 0 <= dimension < x._value.ndim for dimension in dimensions)
    ):
        raise ValueError("rev dimensions must be unique in-bounds axes.")
    if x._privacy_axis in dimensions:
        raise DPError("rev cannot reverse the privacy dimension.")
    return PrivArray(
        value=jax.lax.rev(x._value, dimensions),
        distance=x._distance,
        privacy_axis=x._privacy_axis,
        domain=x._domain,
        parents=[x],
        keep_alignment=True,
    )


@rev.register(remote=False)
def _(x: jax.Array, dimensions: tuple[int, ...]) -> jax.Array:
    return jax.lax.rev(x, dimensions)


def _unary_private(
    x: PrivArray,
    value: jax.Array,
    *,
    domain: ArrayDomain | None = None,
) -> PrivArray:
    return PrivArray(
        value=value,
        distance=x._distance,
        privacy_axis=_preserved_private_axis(
            x,
            value,
            "unary primitive",
        ),
        domain=ArrayDomain() if domain is None else domain,
        parents=[x],
        keep_alignment=True,
    )


def _convert_value(
    x: jax.Array,
    new_dtype: str,
    weak_type: bool,
) -> jax.Array:
    if weak_type:
        raise NotImplementedError(
            "weak convert_element_type outputs are not supported."
        )
    return jax.lax.convert_element_type(x, np.dtype(new_dtype))


@egrpc.multifunction
def convert_element_type(
    x: PrivArray,
    new_dtype: str,
    weak_type: bool,
    sharding: str | None = None,
) -> PrivArray:
    _require_none("convert_element_type sharding", sharding)
    return _unary_private(
        x,
        _convert_value(x._value, new_dtype, weak_type),
        domain=x._domain,
    )


@convert_element_type.register
def _(
    x: SensitiveArray,
    new_dtype: str,
    weak_type: bool,
    sharding: str | None = None,
) -> SensitiveArray:
    _require_none("convert_element_type sharding", sharding)
    return SensitiveArray(
        value=_convert_value(x._value, new_dtype, weak_type),
        distance=x._distance,
        norm_type=x._norm_type,
        parents=[x],
    )


@convert_element_type.register(remote=False)
def _(
    x: jax.Array,
    new_dtype: str,
    weak_type: bool,
    sharding: str | None = None,
) -> jax.Array:
    _require_none("convert_element_type sharding", sharding)
    return _convert_value(x, new_dtype, weak_type)


def _binary_private_public(
    x: PrivArray,
    value: jax.Array,
    primitive: str,
    *,
    domain: ArrayDomain | None = None,
) -> PrivArray:
    return PrivArray(
        value=value,
        distance=x._distance,
        privacy_axis=_preserved_private_axis(x, value, primitive),
        domain=ArrayDomain() if domain is None else domain,
        parents=[x],
        keep_alignment=True,
    )


def _binary_private_private(
    x: PrivArray,
    y: PrivArray,
    value: jax.Array,
    primitive: str,
    *,
    domain: ArrayDomain | None = None,
) -> PrivArray:
    assert_alignment_signature(x, y)
    assert_same_distance_and_accountant(x, y)
    x_rank_offset = value.ndim - x._value.ndim
    y_rank_offset = value.ndim - y._value.ndim
    if x_rank_offset < 0 or y_rank_offset < 0:
        raise DPError(
            f"Private {primitive} cannot remove operand dimensions."
        )
    x_output_axis = x_rank_offset + x._privacy_axis
    y_output_axis = y_rank_offset + y._privacy_axis
    if (
        x_output_axis != y_output_axis
    ):
        raise DPError(
            f"Private {primitive} cannot broadcast or misalign the privacy dimension."
        )
    return PrivArray(
        value=value,
        distance=x._distance,
        privacy_axis=x_output_axis,
        domain=ArrayDomain() if domain is None else domain,
        parents=[x, y],
        keep_alignment=True,
    )


@egrpc.multifunction
def max(x: PrivArray, y: jax.Array) -> PrivArray:
    _assert_public_broadcast(x, y, "max")
    return _binary_private_public(
        x,
        jax.lax.max(x._value, y),
        "max",
    )


@max.register
def _(x: jax.Array, y: PrivArray) -> PrivArray:
    _assert_public_broadcast(y, x, "max")
    return _binary_private_public(
        y,
        jax.lax.max(x, y._value),
        "max",
    )


@max.register
def _(x: PrivArray, y: PrivArray) -> PrivArray:
    return _binary_private_private(
        x,
        y,
        jax.lax.max(x._value, y._value),
        "max",
    )


@max.register(remote=False)
def _(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.max(x, y)


@egrpc.multifunction
def gt(x: PrivArray, y: jax.Array) -> PrivArray:
    _assert_public_broadcast(x, y, "gt")
    return _binary_private_public(
        x,
        jax.lax.gt(x._value, y),
        "gt",
        domain=ArrayDomain(value_range=(0.0, 1.0)),
    )


@gt.register
def _(x: jax.Array, y: PrivArray) -> PrivArray:
    _assert_public_broadcast(y, x, "gt")
    return _binary_private_public(
        y,
        jax.lax.gt(x, y._value),
        "gt",
        domain=ArrayDomain(value_range=(0.0, 1.0)),
    )


@gt.register
def _(x: PrivArray, y: PrivArray) -> PrivArray:
    return _binary_private_private(
        x,
        y,
        jax.lax.gt(x._value, y._value),
        "gt",
        domain=ArrayDomain(value_range=(0.0, 1.0)),
    )


@gt.register(remote=False)
def _(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.gt(x, y)


@egrpc.multifunction
def lt(x: PrivArray, y: jax.Array) -> PrivArray:
    _assert_public_broadcast(x, y, "lt")
    return _binary_private_public(
        x,
        jax.lax.lt(x._value, y),
        "lt",
        domain=ArrayDomain(value_range=(0.0, 1.0)),
    )


@lt.register
def _(x: jax.Array, y: PrivArray) -> PrivArray:
    _assert_public_broadcast(y, x, "lt")
    return _binary_private_public(
        y,
        jax.lax.lt(x, y._value),
        "lt",
        domain=ArrayDomain(value_range=(0.0, 1.0)),
    )


@lt.register
def _(x: PrivArray, y: PrivArray) -> PrivArray:
    return _binary_private_private(
        x,
        y,
        jax.lax.lt(x._value, y._value),
        "lt",
        domain=ArrayDomain(value_range=(0.0, 1.0)),
    )


@lt.register(remote=False)
def _(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.lt(x, y)


@egrpc.multifunction
def div(x: PrivArray, y: jax.Array) -> PrivArray:
    _assert_public_broadcast(x, y, "div")
    return _binary_private_public(
        x,
        jax.lax.div(x._value, y),
        "div",
    )


@div.register
def _(x: jax.Array, y: PrivArray) -> PrivArray:
    _assert_public_broadcast(y, x, "div")
    return _binary_private_public(
        y,
        jax.lax.div(x, y._value),
        "div",
    )


@div.register
def _(x: PrivArray, y: PrivArray) -> PrivArray:
    return _binary_private_private(
        x,
        y,
        jax.lax.div(x._value, y._value),
        "div",
    )


@div.register(remote=False)
def _(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.lax.div(x, y)


@egrpc.multifunction
def stop_gradient(x: PrivArray) -> PrivArray:
    return _unary_private(
        x,
        jax.lax.stop_gradient(x._value),
        domain=x._domain,
    )


@stop_gradient.register(remote=False)
def _(x: jax.Array) -> jax.Array:
    return jax.lax.stop_gradient(x)


@egrpc.multifunction
def exp(
    x: PrivArray,
    accuracy: str | None = None,
) -> PrivArray:
    _require_none("exp accuracy", accuracy)
    return _unary_private(x, jax.lax.exp(x._value))


@exp.register(remote=False)
def _(
    x: jax.Array,
    accuracy: str | None = None,
) -> jax.Array:
    _require_none("exp accuracy", accuracy)
    return jax.lax.exp(x)


@egrpc.multifunction
def log(
    x: PrivArray,
    accuracy: str | None = None,
) -> PrivArray:
    _require_none("log accuracy", accuracy)
    return _unary_private(x, jax.lax.log(x._value))


@log.register(remote=False)
def _(
    x: jax.Array,
    accuracy: str | None = None,
) -> jax.Array:
    _require_none("log accuracy", accuracy)
    return jax.lax.log(x)


@egrpc.multifunction
def neg(x: PrivArray) -> PrivArray:
    return _unary_private(
        x,
        jax.lax.neg(x._value),
        domain=ArrayDomain(
            norm_type=x._domain.norm_type,
            norm_bound=x._domain.norm_bound,
            value_range=_scale_value_range(
                x._domain.value_range,
                -1.0,
            ),
        ),
    )


@neg.register(remote=False)
def _(x: jax.Array) -> jax.Array:
    return jax.lax.neg(x)


@egrpc.multifunction
def select_n(which: PrivArray, cases: list[PrivArray]) -> PrivArray:
    if not cases:
        raise ValueError("select_n requires at least one case.")
    assert_privacy_axis(which, *cases)
    assert_same_distance_and_accountant(which, *cases)
    if any(case._value.shape != which._value.shape for case in cases):
        raise DPError("Private select_n requires equal, non-broadcasting shapes.")
    return PrivArray(
        value=jax.lax.select_n(
            which._value,
            *(case._value for case in cases),
        ),
        distance=which._distance,
        privacy_axis=which._privacy_axis,
        domain=ArrayDomain(),
        parents=[which, *cases],
        keep_alignment=True,
    )


@select_n.register(remote=False)
def _(which: jax.Array, cases: list[jax.Array]) -> jax.Array:
    return jax.lax.select_n(which, *cases)


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    normalized = tuple(axis + ndim if axis < 0 else axis for axis in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Reduction axes must be unique.")
    if any(not 0 <= axis < ndim for axis in normalized):
        raise ValueError("Reduction axis is out of bounds.")
    return tuple(sorted(normalized))


@egrpc.multifunction
def reduce_max(x: PrivArray, axes: tuple[int, ...]) -> PrivArray:
    normalized = _normalize_axes(axes, x._value.ndim)
    if x._privacy_axis in normalized:
        raise DPError("reduce_max cannot reduce the privacy dimension.")
    return PrivArray(
        value=jax.lax.reduce_max(x._value, normalized),
        distance=x._distance,
        privacy_axis=x._privacy_axis
        - sum(axis < x._privacy_axis for axis in normalized),
        domain=ArrayDomain(value_range=x._domain.value_range),
        parents=[x],
        keep_alignment=True,
    )


@reduce_max.register(remote=False)
def _(x: jax.Array, axes: tuple[int, ...]) -> jax.Array:
    return jax.lax.reduce_max(x, _normalize_axes(axes, x.ndim))


@egrpc.multifunction
def squeeze(x: PrivArray, dimensions: tuple[int, ...]) -> PrivArray:
    normalized = _normalize_axes(dimensions, x._value.ndim)
    if x._privacy_axis in normalized:
        raise DPError("squeeze cannot remove the privacy dimension.")
    if any(x._value.shape[axis] != 1 for axis in normalized):
        raise ValueError("squeeze dimensions must have size one.")
    return PrivArray(
        value=jax.lax.squeeze(x._value, normalized),
        distance=x._distance,
        privacy_axis=x._privacy_axis
        - sum(axis < x._privacy_axis for axis in normalized),
        domain=x._domain,
        parents=[x],
        keep_alignment=True,
    )


@squeeze.register(remote=False)
def _(x: jax.Array, dimensions: tuple[int, ...]) -> jax.Array:
    return jax.lax.squeeze(x, _normalize_axes(dimensions, x.ndim))


# egrpc wire form; the trusted endpoint reconstructs JAX's named tuple.
GatherDimensionNumbersParam = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]


def _gather_dimensions(
    dimension_numbers: GatherDimensionNumbersParam,
) -> jax.lax.GatherDimensionNumbers:
    (
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        operand_batching_dims,
        start_indices_batching_dims,
    ) = dimension_numbers
    return jax.lax.GatherDimensionNumbers(
        offset_dims=offset_dims,
        collapsed_slice_dims=collapsed_slice_dims,
        start_index_map=start_index_map,
        operand_batching_dims=operand_batching_dims,
        start_indices_batching_dims=start_indices_batching_dims,
    )


def _gather_value(
    operand: jax.Array,
    start_indices: jax.Array,
    *,
    dimension_numbers: GatherDimensionNumbersParam,
    slice_sizes: tuple[int, ...],
    unique_indices: bool,
    indices_are_sorted: bool,
    mode: str,
    fill_value: bool | int | float | None,
) -> jax.Array:
    if fill_value is not None:
        raise NotImplementedError("gather fill_value is not supported.")
    return jax.lax.gather(
        operand,
        start_indices,
        _gather_dimensions(dimension_numbers),
        slice_sizes,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        mode=getattr(jax.lax.GatherScatterMode, mode),
        fill_value=None,
    )


def _has_recordwise_private_batching(
    operand_batching_dims: tuple[int, ...],
    indices_batching_dims: tuple[int, ...],
) -> bool:
    if len(operand_batching_dims) != len(indices_batching_dims):
        return False
    try:
        private_position = operand_batching_dims.index(0)
    except ValueError:
        return False
    return indices_batching_dims[private_position] == 0


@egrpc.multifunction
def gather(
    operand: PrivArray,
    start_indices: PrivArray,
    dimension_numbers: GatherDimensionNumbersParam,
    slice_sizes: tuple[int, ...],
    unique_indices: bool,
    indices_are_sorted: bool,
    mode: str,
    fill_value: bool | int | float | None,
) -> PrivArray:
    operand_batching_dims = dimension_numbers[3]
    start_indices_batching_dims = dimension_numbers[4]
    assert_privacy_axis(operand, start_indices)
    assert_same_distance_and_accountant(operand, start_indices)
    if (
        operand._privacy_axis != 0
        or not _has_recordwise_private_batching(
            operand_batching_dims,
            start_indices_batching_dims,
        )
        or slice_sizes[operand._privacy_axis] != 1
    ):
        raise DPError(
            "The prototype only supports record-wise batched gather at axis 0."
        )
    value = _gather_value(
        operand._value,
        start_indices._value,
        dimension_numbers=dimension_numbers,
        slice_sizes=slice_sizes,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        mode=mode,
        fill_value=fill_value,
    )
    if value.ndim == 0 or value.shape[0] != operand._value.shape[0]:
        raise DPError("gather must preserve the private batch as output axis 0.")
    return PrivArray(
        value=value,
        distance=operand._distance,
        privacy_axis=0,
        domain=ArrayDomain(),
        parents=[operand, start_indices],
        keep_alignment=True,
    )


@gather.register(remote=False)
def _(
    operand: jax.Array,
    start_indices: jax.Array,
    dimension_numbers: GatherDimensionNumbersParam,
    slice_sizes: tuple[int, ...],
    unique_indices: bool,
    indices_are_sorted: bool,
    mode: str,
    fill_value: bool | int | float | None,
) -> jax.Array:
    return _gather_value(
        operand,
        start_indices,
        dimension_numbers=dimension_numbers,
        slice_sizes=slice_sizes,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        mode=mode,
        fill_value=fill_value,
    )


# egrpc wire form; the trusted endpoint reconstructs JAX's named tuple.
ScatterDimensionNumbersParam = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]


def _scatter_dimensions(
    dimension_numbers: ScatterDimensionNumbersParam,
) -> jax.lax.ScatterDimensionNumbers:
    (
        update_window_dims,
        inserted_window_dims,
        scatter_dims_to_operand_dims,
        operand_batching_dims,
        scatter_indices_batching_dims,
    ) = dimension_numbers
    return jax.lax.ScatterDimensionNumbers(
        update_window_dims=update_window_dims,
        inserted_window_dims=inserted_window_dims,
        scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
        operand_batching_dims=operand_batching_dims,
        scatter_indices_batching_dims=scatter_indices_batching_dims,
    )


def _scatter_value(
    operand: jax.Array,
    scatter_indices: jax.Array,
    updates: jax.Array,
    *,
    dimension_numbers: ScatterDimensionNumbersParam,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str,
    update_jaxpr: str | None,
    update_consts: bool,
) -> jax.Array:
    if update_jaxpr is not None:
        raise NotImplementedError(
            "scatter update computations are not supported."
        )
    if update_consts:
        raise NotImplementedError("scatter update constants are not supported.")
    return jax.lax.scatter(
        operand,
        scatter_indices,
        updates,
        _scatter_dimensions(dimension_numbers),
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=getattr(jax.lax.GatherScatterMode, mode),
    )


def _scatter_add_value(
    operand: jax.Array,
    scatter_indices: jax.Array,
    updates: jax.Array,
    *,
    dimension_numbers: ScatterDimensionNumbersParam,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str,
    update_jaxpr: str | None,
    update_consts: bool,
) -> jax.Array:
    if update_jaxpr != "add":
        raise NotImplementedError(
            "scatter-add requires its addition computation."
        )
    if update_consts:
        raise NotImplementedError(
            "scatter-add update constants are not supported."
        )
    return jax.lax.scatter_add(
        operand,
        scatter_indices,
        updates,
        _scatter_dimensions(dimension_numbers),
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=getattr(jax.lax.GatherScatterMode, mode),
    )


@egrpc.multifunction
def scatter(
    operand: PrivArray,
    scatter_indices: PrivArray,
    updates: PrivArray,
    dimension_numbers: ScatterDimensionNumbersParam,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str,
    update_jaxpr: str | None,
    update_consts: bool,
) -> PrivArray:
    assert_privacy_axis(operand, scatter_indices, updates)
    assert_same_distance_and_accountant(
        operand,
        scatter_indices,
        updates,
    )
    if (
        operand._privacy_axis != 0
        or not _has_recordwise_private_batching(
            dimension_numbers[3],
            dimension_numbers[4],
        )
    ):
        raise DPError(
            "The prototype only supports record-wise batched scatter at axis 0."
        )
    value = _scatter_value(
        operand._value,
        scatter_indices._value,
        updates._value,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
    )
    if value.shape != operand._value.shape:
        raise DPError("scatter must preserve the complete private operand shape.")
    return PrivArray(
        value=value,
        distance=operand._distance,
        privacy_axis=0,
        domain=ArrayDomain(),
        parents=[operand, scatter_indices, updates],
        keep_alignment=True,
    )


@scatter.register(remote=False)
def _(
    operand: jax.Array,
    scatter_indices: jax.Array,
    updates: jax.Array,
    dimension_numbers: ScatterDimensionNumbersParam,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str,
    update_jaxpr: str | None,
    update_consts: bool,
) -> jax.Array:
    return _scatter_value(
        operand,
        scatter_indices,
        updates,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
    )


@egrpc.multifunction
def scatter_add(
    operand: PrivArray,
    scatter_indices: PrivArray,
    updates: PrivArray,
    dimension_numbers: ScatterDimensionNumbersParam,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str,
    update_jaxpr: str | None,
    update_consts: bool,
) -> PrivArray:
    assert_privacy_axis(operand, scatter_indices, updates)
    assert_same_distance_and_accountant(
        operand,
        scatter_indices,
        updates,
    )
    if (
        operand._privacy_axis != 0
        or not _has_recordwise_private_batching(
            dimension_numbers[3],
            dimension_numbers[4],
        )
    ):
        raise DPError(
            "The prototype only supports record-wise batched scatter at axis 0."
        )
    value = _scatter_add_value(
        operand._value,
        scatter_indices._value,
        updates._value,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
    )
    if value.shape != operand._value.shape:
        raise DPError("scatter must preserve the complete private operand shape.")
    return PrivArray(
        value=value,
        distance=operand._distance,
        privacy_axis=0,
        domain=ArrayDomain(),
        parents=[operand, scatter_indices, updates],
        keep_alignment=True,
    )


@scatter_add.register(remote=False)
def _(
    operand: jax.Array,
    scatter_indices: jax.Array,
    updates: jax.Array,
    dimension_numbers: ScatterDimensionNumbersParam,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str,
    update_jaxpr: str | None,
    update_consts: bool,
) -> jax.Array:
    return _scatter_add_value(
        operand,
        scatter_indices,
        updates,
        dimension_numbers=dimension_numbers,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
        mode=mode,
        update_jaxpr=update_jaxpr,
        update_consts=update_consts,
    )


@egrpc.multifunction
def reduce_sum(
    x: PrivArray,
    axes: tuple[int, ...],
    out_sharding: str | None = None,
) -> PrivArray | SensitiveArray:
    _require_none("reduce_sum out_sharding", out_sharding)
    normalized = _normalize_axes(axes, x._value.ndim)
    result = jax.lax.reduce_sum(x._value, normalized)

    if x._privacy_axis in normalized:
        if normalized != (x._privacy_axis,):
            raise NotImplementedError(
                "The prototype reduces the privacy axis separately from other axes."
            )
        norm_bound = x._domain.norm_bound
        if norm_bound is None:
            raise DPError(
                "Norm bound is not set. Clip before reducing the privacy axis."
            )
        return SensitiveArray(
            value=result,
            distance=x._distance * norm_bound,
            norm_type=x._domain.norm_type,
            parents=[x],
        )

    new_privacy_axis = x._privacy_axis - sum(
        axis < x._privacy_axis for axis in normalized
    )
    value_range = x._domain.value_range
    if value_range is not None:
        count = int(np.prod([x._value.shape[axis] for axis in normalized]))
        lo, hi = value_range
        value_range = (
            lo * count if lo is not None else None,
            hi * count if hi is not None else None,
        )

    return PrivArray(
        value=result,
        distance=x._distance,
        privacy_axis=new_privacy_axis,
        domain=ArrayDomain(value_range=value_range),
        parents=[x],
        keep_alignment=True,
    )


@reduce_sum.register(remote=False)
def _(
    x: jax.Array,
    axes: tuple[int, ...],
    out_sharding: str | None = None,
) -> jax.Array:
    _require_none("reduce_sum out_sharding", out_sharding)
    return jax.lax.reduce_sum(x, _normalize_axes(axes, x.ndim))


def as_jax_value(value: PrimitiveValue) -> jax.Array:
    if isinstance(value, (PrivArray, SensitiveArray)):
        return value._value
    return value
