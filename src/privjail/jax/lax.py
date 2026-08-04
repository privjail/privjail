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

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from ..array_base import SensitiveDimInt
from ..util import DPError
from .array import PrivArray
from .primitives import conv_general_dilated as _trusted_conv_general_dilated


def _shape(
    array: PrivArray | jax.Array,
) -> tuple[int | SensitiveDimInt, ...]:
    if isinstance(array, PrivArray):
        return array.shape
    return tuple(int(dim) for dim in array.shape)


def _concrete_spatial_shape(
    shape: tuple[int | SensitiveDimInt, ...],
    axes: tuple[int, ...],
    operand: str,
) -> tuple[int, ...]:
    result: list[int] = []
    for axis in axes:
        dimension = shape[axis]
        if isinstance(dimension, SensitiveDimInt):
            raise DPError(
                f"{operand} convolution spatial dimensions cannot be private."
            )
        result.append(dimension)
    return tuple(result)


def conv_general_dilated(
    lhs: PrivArray | jax.Array,
    rhs: PrivArray | jax.Array,
    window_strides: Sequence[int],
    padding: str | Sequence[tuple[int, int]],
    lhs_dilation: Sequence[int] | None = None,
    rhs_dilation: Sequence[int] | None = None,
    dimension_numbers: Any = None,
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    precision: Any = None,
    preferred_element_type: Any | None = None,
    out_sharding: Any | None = None,
) -> PrivArray | jax.Array:
    if not isinstance(lhs, PrivArray) and not isinstance(rhs, PrivArray):
        return jax.lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides,
            padding,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
            batch_group_count=batch_group_count,
            precision=precision,
            preferred_element_type=preferred_element_type,
            out_sharding=out_sharding,
        )
    if not isinstance(lhs, PrivArray):
        raise NotImplementedError(
            "A private convolution rhs requires a private lhs."
        )
    if precision is not None:
        raise NotImplementedError(
            "Private conv_general_dilated precision is not supported."
        )
    if out_sharding is not None:
        raise NotImplementedError(
            "Private conv_general_dilated out_sharding is not supported."
        )

    lhs_shape = _shape(lhs)
    rhs_shape = _shape(rhs)
    dimensions = jax.lax.conv_dimension_numbers(
        (1,) * len(lhs_shape),
        (1,) * len(rhs_shape),
        dimension_numbers,
    )
    strides = tuple(int(stride) for stride in window_strides)
    if len(strides) != 3:
        raise NotImplementedError(
            "PrivJail currently supports only 3D convolution."
        )
    lhs_rates = (
        (1,) * 3
        if lhs_dilation is None
        else tuple(int(rate) for rate in lhs_dilation)
    )
    rhs_rates = (
        (1,) * 3
        if rhs_dilation is None
        else tuple(int(rate) for rate in rhs_dilation)
    )
    if isinstance(padding, str):
        if any(rate != 1 for rate in lhs_rates):
            raise ValueError(
                "String padding is not supported with lhs dilation."
            )
        lhs_spatial = _concrete_spatial_shape(
            lhs_shape,
            tuple(dimensions.lhs_spec[2:]),
            "lhs",
        )
        rhs_spatial = _concrete_spatial_shape(
            rhs_shape,
            tuple(dimensions.rhs_spec[2:]),
            "rhs",
        )
        effective_rhs = tuple(
            (size - 1) * rate + 1
            for size, rate in zip(
                rhs_spatial,
                rhs_rates,
                strict=True,
            )
        )
        pads = tuple(
            jax.lax.padtype_to_pads(
                lhs_spatial,
                effective_rhs,
                strides,
                padding,
            )
        )
    else:
        pads = tuple((int(low), int(high)) for low, high in padding)

    result: PrivArray = _trusted_conv_general_dilated(
        lhs,
        rhs,
        strides,
        pads,
        lhs_rates,
        rhs_rates,
        (
            tuple(dimensions.lhs_spec),
            tuple(dimensions.rhs_spec),
            tuple(dimensions.out_spec),
        ),
        int(feature_group_count),
        int(batch_group_count),
        None,
        (
            None
            if preferred_element_type is None
            else str(jnp.dtype(preferred_element_type))
        ),
        None,
    )
    return result


__all__ = ["conv_general_dilated"]
