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

from dataclasses import fields
from enum import Enum
import re
from types import UnionType
from typing import Any, Callable, Union, get_args, get_origin

import egrpc
import jax
import jax.extend.core as jax_core
import jax.numpy as jnp
import numpy as np

from ..array_base import SensitiveDimInt
from ..util import DPError
from .call_batch import (
    InputValue,
    ResultValue,
    get_output,
    pack_outputs,
    primitive_endpoint,
    privacy_dimension,
    public_constant,
    scale_dimension,
)


DimensionValue = int | SensitiveDimInt
_SYMBOLIC_DIMENSION = re.compile(
    r"(?:(?P<factor>[1-9][0-9]*)\*)?"
    r"(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\Z"
)


def _parse_symbolic_dimension(dimension: str) -> tuple[str, int]:
    match = _SYMBOLIC_DIMENSION.fullmatch(dimension)
    if match is None:
        raise ValueError(f"Invalid symbolic dimension: {dimension!r}")
    factor_text = match.group("factor")
    return match.group("symbol"), 1 if factor_text is None else int(factor_text)


def _update_jaxpr_name(value: Any) -> str | None:
    if value is None:
        return None
    jaxpr = value.jaxpr if isinstance(value, jax_core.ClosedJaxpr) else value
    if not isinstance(jaxpr, jax_core.Jaxpr):
        raise NotImplementedError(
            f"Unsupported update_jaxpr: {type(value).__name__}"
        )
    if (
        jaxpr.constvars
        or jaxpr.effects
        or len(jaxpr.eqns) != 1
        or jaxpr.eqns[0].primitive.name != "add"
    ):
        raise NotImplementedError("Only the scatter-add update JAXPR is supported.")
    return "add"


def _dimension(dimension: Any) -> int | str:
    if isinstance(dimension, (int, np.integer)):
        return int(dimension)
    result = str(dimension)
    _parse_symbolic_dimension(result)
    return result


def _normalize_param(name: str, value: Any) -> Any:
    if name in {"shape", "new_sizes"}:
        return tuple(_dimension(dimension) for dimension in value)
    if name in {"feature_group_count", "batch_group_count"}:
        return _dimension(value)
    if name == "update_jaxpr":
        return _update_jaxpr_name(value)
    if name == "update_consts":
        return bool(value)
    if name in {"accuracy", "precision", "sharding", "out_sharding"}:
        return None if value is None else str(value)
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return tuple(_normalize_param(name, item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise NotImplementedError(
        f"Unsupported JAX parameter {name}: {value!r}"
    )


def _contains_list(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return any(_contains_list(argument) for argument in get_args(annotation))
    return origin is list


def _bind_primitive(
    name: str,
    operands: list[Any],
    params: dict[str, Any],
    resolve_dimension: Callable[[int | str], DimensionValue],
) -> Any:
    endpoint = primitive_endpoint(name)
    call_fields = fields(egrpc.call_type(endpoint))
    list_fields = [
        index
        for index, field in enumerate(call_fields)
        if _contains_list(field.type)
    ]
    if len(list_fields) > 1:
        raise RuntimeError(
            f"{name} has more than one variadic operand field."
        )
    if list_fields and any(
        field.name not in params
        for field in call_fields[list_fields[0] + 1 :]
    ):
        raise ValueError(
            f"{name} variadic operands must be followed only by "
            "explicit parameters."
        )

    arguments: dict[str, Any] = {}
    operand_index = 0
    for field in call_fields:
        if operand_index == len(operands):
            break
        if field.name in params:
            raise ValueError(
                f"{name} has too many operands for its endpoint signature."
            )
        if _contains_list(field.type):
            arguments[field.name] = operands[operand_index:]
            operand_index = len(operands)
        else:
            arguments[field.name] = operands[operand_index]
            operand_index += 1
    if operand_index != len(operands):
        raise ValueError(
            f"{name} operand arity does not match its endpoint signature."
        )

    field_names = {field.name for field in call_fields}
    unknown = set(params) - field_names
    if unknown:
        raise NotImplementedError(
            f"Unsupported {name} parameters: {sorted(unknown)!r}"
        )
    normalized = {
        param: _normalize_param(param, value)
        for param, value in params.items()
    }
    for parameter_name in ("shape", "new_sizes"):
        if parameter_name in normalized:
            normalized[parameter_name] = tuple(
                resolve_dimension(dimension)
                for dimension in normalized[parameter_name]
            )
    for parameter_name in ("feature_group_count", "batch_group_count"):
        if parameter_name in normalized:
            normalized[parameter_name] = resolve_dimension(
                normalized[parameter_name]
            )
    arguments.update(normalized)
    try:
        result = endpoint(**arguments)
    except TypeError as exc:
        raise NotImplementedError(
            f"{name} parameters do not match its endpoint signature."
        ) from exc
    return result


def _normalize_constant(value: Any) -> jax.Array:
    if isinstance(value, jax.Array):
        return value

    array = np.asarray(value)
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"Unsupported constant dtype: {array.dtype}")
    return jnp.asarray(value)


def lower_jaxpr(
    closed_jaxpr: jax_core.ClosedJaxpr,
    inputs: list[InputValue],
) -> egrpc.CallBatch:
    if len(inputs) != len(closed_jaxpr.jaxpr.invars):
        raise ValueError("JAXPR input arity does not match its call.")

    def lower(*argument_values: Any) -> list[ResultValue]:
        dimension_refs: dict[str, SensitiveDimInt] = {}
        scaled_dimensions: dict[tuple[str, int], SensitiveDimInt] = {}
        for var, value in zip(
            closed_jaxpr.jaxpr.invars,
            argument_values,
            strict=True,
        ):
            for axis, dimension in enumerate(var.aval.shape):
                if isinstance(dimension, (int, np.integer)):
                    continue
                symbol, factor = _parse_symbolic_dimension(str(dimension))
                if factor != 1:
                    raise NotImplementedError(
                        "A JAX input cannot use a scaled symbolic dimension."
                    )
                if symbol not in dimension_refs:
                    dimension_refs[symbol] = privacy_dimension(
                        value=value,
                        axis=axis,
                    )

        def resolve_dimension(dimension: int | str) -> DimensionValue:
            if isinstance(dimension, int):
                return dimension
            symbol, factor = _parse_symbolic_dimension(dimension)
            try:
                reference = dimension_refs[symbol]
            except KeyError as exc:
                raise DPError(
                    f"Symbolic dimension {symbol} has no private input."
                ) from exc
            if factor == 1:
                return reference
            key = (symbol, factor)
            if key not in scaled_dimensions:
                scaled_dimensions[key] = scale_dimension(
                    dimension=reference,
                    factor=factor,
                )
            return scaled_dimensions[key]

        def resolve_atom(
            atom: Any,
            environment: dict[Any, Any],
        ) -> Any:
            if isinstance(atom, jax_core.Literal):
                return public_constant(
                    value=_normalize_constant(atom.val),
                )
            if isinstance(atom, jax_core.Var):
                try:
                    return environment[atom]
                except KeyError as exc:
                    raise ValueError(
                        "JAXPR uses a variable before it is defined."
                    ) from exc
            raise TypeError(f"Unsupported JAXPR atom: {atom!r}")

        def nested_call(
            primitive: str,
            params: dict[str, Any],
        ) -> jax_core.ClosedJaxpr | None:
            if primitive == "jit":
                nested = params.get("jaxpr")
            elif primitive == "custom_jvp_call":
                nested = params.get("call_jaxpr")
            else:
                return None
            if not isinstance(nested, jax_core.ClosedJaxpr):
                raise TypeError(f"{primitive} does not contain a ClosedJaxpr.")
            return nested

        def convert(
            current: jax_core.ClosedJaxpr,
            input_values: list[Any],
        ) -> list[Any]:
            jaxpr = current.jaxpr
            if jaxpr.effects:
                raise NotImplementedError(
                    f"JAX effects are not supported: {jaxpr.effects!r}"
                )
            if len(input_values) != len(jaxpr.invars):
                raise ValueError(
                    "Nested JAXPR input arity does not match its call."
                )

            environment = dict(zip(jaxpr.invars, input_values, strict=True))
            for var, value in zip(jaxpr.constvars, current.consts, strict=True):
                environment[var] = public_constant(
                    value=_normalize_constant(value),
                )

            for equation in jaxpr.eqns:
                name = equation.primitive.name
                if equation.effects:
                    raise NotImplementedError(
                        f"JAX primitive {name} has unsupported effects."
                    )
                operands = [
                    resolve_atom(atom, environment)
                    for atom in equation.invars
                ]
                nested = nested_call(name, equation.params)
                if nested is not None:
                    nested_outputs = convert(nested, operands)
                    if len(nested_outputs) != len(equation.outvars):
                        raise ValueError(
                            f"{name} output arity does not match its nested JAXPR."
                        )
                    environment.update(
                        zip(equation.outvars, nested_outputs, strict=True)
                    )
                    continue

                result = _bind_primitive(
                    name,
                    operands,
                    equation.params,
                    resolve_dimension,
                )
                if equation.primitive.multiple_results:
                    for index, var in enumerate(equation.outvars):
                        environment[var] = get_output(
                            values=result,
                            index=index,
                        )
                else:
                    if len(equation.outvars) != 1:
                        raise ValueError(
                            f"{name} must produce exactly one JAXPR output."
                        )
                    environment[equation.outvars[0]] = result

            return [
                resolve_atom(atom, environment)
                for atom in jaxpr.outvars
            ]

        outputs = convert(closed_jaxpr, list(argument_values))
        return pack_outputs(values=outputs)

    return egrpc.trace(lower, *inputs)


__all__ = ["lower_jaxpr"]
