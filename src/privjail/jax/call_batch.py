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

from dataclasses import fields, is_dataclass
from functools import cache
from typing import Any, Callable

import egrpc
import jax

from ..array_base import SensitiveDimInt
from ..prisoner import Prisoner
from ..util import DPError
from .array import PrivArray, SensitiveArray


InputValue = PrivArray | SensitiveArray | jax.Array | bool | int | float
ResultValue = PrivArray | SensitiveArray | jax.Array
PrimitiveEndpoint = Callable[..., Any]


@egrpc.function
def public_constant(value: jax.Array) -> jax.Array:
    return value


@egrpc.function
def privacy_dimension(value: PrivArray, axis: int) -> SensitiveDimInt:
    dimension = value.shape[axis]
    if isinstance(dimension, int):
        raise DPError("The selected JAX argument axis is public.")
    return dimension


@egrpc.function
def scale_dimension(
    dimension: SensitiveDimInt,
    factor: int,
) -> SensitiveDimInt:
    if factor <= 0:
        raise ValueError("A symbolic dimension factor must be positive.")
    result: SensitiveDimInt = dimension * factor
    return result


@egrpc.multifunction
def get_output(values: list[PrivArray], index: int) -> PrivArray:
    return values[index]


@get_output.register
def _(values: list[SensitiveArray], index: int) -> SensitiveArray:
    return values[index]


@get_output.register(remote=False)
def _(values: list[jax.Array], index: int) -> jax.Array:
    return values[index]


@egrpc.function
def pack_outputs(values: list[ResultValue]) -> list[ResultValue]:
    return values


@cache
def _primitive_endpoints() -> dict[str, PrimitiveEndpoint]:
    from . import helper, mechanism, primitives

    endpoints: dict[str, PrimitiveEndpoint] = {}
    for module in (primitives, helper, mechanism):
        for name, value in vars(module).items():
            if name.startswith("_") or not callable(value):
                continue
            try:
                egrpc.call_types(value)
            except TypeError:
                continue
            primitive = "scatter-add" if name == "scatter_add" else name
            if primitive in endpoints:
                raise RuntimeError(f"Duplicate JAX primitive endpoint: {primitive}")
            endpoints[primitive] = value
    return endpoints


def primitive_endpoint(name: str) -> PrimitiveEndpoint:
    try:
        return _primitive_endpoints()[name]
    except KeyError as exc:
        raise NotImplementedError(
            f"Unsupported JAX primitive: {name}"
        ) from exc


@cache
def _call_names() -> dict[type[egrpc.Call], str]:
    return {
        call_type: name
        for name, function in _primitive_endpoints().items()
        for call_type in egrpc.call_types(function)
    }


def primitive_name(call: egrpc.Call) -> str | None:
    return _call_names().get(type(call))


def _children(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [*value.keys(), *value.values()]
    if is_dataclass(value) and not isinstance(value, type):
        return [getattr(value, field.name) for field in fields(value)]
    return []


def _references(value: Any) -> list[egrpc.ValueRef]:
    if isinstance(value, egrpc.ValueRef):
        return [value]
    return [
        reference
        for child in _children(value)
        for reference in _references(child)
    ]


def _contains_protected_value(value: Any) -> bool:
    return isinstance(value, Prisoner) or any(
        _contains_protected_value(child)
        for child in _children(value)
    )


def validate_call_batch(batch: egrpc.CallBatch) -> int:
    if batch.output is None:
        raise ValueError("A JAX CallBatch must have an output.")
    if not batch.calls:
        raise ValueError("A JAX CallBatch must contain calls.")
    if len(batch.calls) > 10_000:
        raise ValueError("A JAX CallBatch contains too many calls.")
    if batch.output.index != batch.input_count + len(batch.calls) - 1:
        raise ValueError("A JAX CallBatch must output its final call.")
    if type(batch.calls[-1]) is not egrpc.call_type(pack_outputs):
        raise TypeError("A JAX CallBatch must end with pack_outputs.")

    support_functions = (
        public_constant,
        privacy_dimension,
        scale_dimension,
        get_output,
        pack_outputs,
    )
    supported_types = set(_call_names()) | {
        call_type
        for function in support_functions
        for call_type in egrpc.call_types(function)
    }
    for index, call in enumerate(batch.calls):
        if type(call) not in supported_types:
            raise NotImplementedError(
                f"Unsupported call in JAX CallBatch: {type(call).__qualname__}"
            )
        if _contains_protected_value(call):
            raise DPError(
                "Protected values must enter a JAX CallBatch through inputs."
            )
        for reference in _references(call):
            if (
                reference.index < 0
                or reference.index >= batch.input_count + index
            ):
                raise ValueError(
                    f"Call {index} contains an invalid ValueRef."
                )

    return batch.input_count


__all__ = [
    "InputValue",
    "ResultValue",
    "get_output",
    "pack_outputs",
    "primitive_endpoint",
    "primitive_name",
    "privacy_dimension",
    "public_constant",
    "scale_dimension",
    "validate_call_batch",
]
