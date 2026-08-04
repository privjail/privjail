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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import update_wrapper
import inspect
from typing import Any, Generic, ParamSpec, TypeVar, cast, overload

import jax
import jax.numpy as jnp
import numpy as np

from ..alignment import AlignmentSignature
from ..prisoner import Prisoner
from .array import PrivArray, SensitiveArray
from .call_batch import InputValue
from .execution import JitComputation, jit_computation
from .lowering import lower_jaxpr


P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class _CachedComputation:
    computation: JitComputation
    out_tree: jax.tree_util.PyTreeDef


def _normalize_index(index: int, count: int) -> int:
    normalized = index + count if index < 0 else index
    if not 0 <= normalized < count:
        raise ValueError(f"static_argnums index {index} is out of range.")
    return normalized


def _static_indices(
    signature: inspect.Signature,
    static_argnums: int | Sequence[int],
    static_argnames: str | Iterable[str],
) -> tuple[int, ...]:
    parameters = tuple(signature.parameters.values())
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        raise NotImplementedError(
            "*args and **kwargs are not supported by the prototype."
        )

    argnums = (static_argnums,) if isinstance(static_argnums, int) else static_argnums
    argnames = (
        (static_argnames,) if isinstance(static_argnames, str) else static_argnames
    )
    indices = {_normalize_index(index, len(parameters)) for index in argnums}
    names = {parameter.name: index for index, parameter in enumerate(parameters)}
    for name in argnames:
        if name not in names:
            raise ValueError(f"Unknown static_argnames entry: {name!r}")
        indices.add(names[name])
    return tuple(sorted(indices))


def _call_ordered(
    function: Callable[..., Any],
    parameters: tuple[inspect.Parameter, ...],
    values: tuple[Any, ...],
) -> Any:
    positional: list[Any] = []
    keywords: dict[str, Any] = {}
    for parameter, value in zip(parameters, values, strict=True):
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional.append(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keywords[parameter.name] = value
        else:
            raise NotImplementedError("*args and **kwargs are not supported.")
    return function(*positional, **keywords)


def _assert_static_value(value: Any) -> None:
    if any(isinstance(leaf, Prisoner) for leaf in jax.tree.leaves(value)):
        raise TypeError("Private and sensitive objects cannot be static arguments.")
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("Static arguments must be hashable.") from exc


def _public_abstract(value: Any) -> Any:
    if isinstance(value, jax.Array):
        return jax.ShapeDtypeStruct(  # type: ignore[no-untyped-call]
            value.shape,
            value.dtype,
            weak_type=bool(getattr(value, "weak_type", False)),
        )
    if isinstance(value, np.ndarray):
        return jax.ShapeDtypeStruct(  # type: ignore[no-untyped-call]
            value.shape,
            value.dtype,
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, complex)):
        return value
    raise TypeError(f"Unsupported dynamic JAX argument leaf: {type(value).__name__}")


def _symbolic_abstract(
    value: PrivArray | SensitiveArray,
    scope: jax.export.SymbolicScope,
    alignment_symbols: dict[AlignmentSignature, str],
) -> Any:
    shape: list[Any] = []
    for dim in value.shape:
        if isinstance(dim, int):
            shape.append(dim)
        else:
            signature = dim.alignment_signature
            symbol = alignment_symbols.setdefault(
                signature,
                f"b{len(alignment_symbols)}",
            )
            expression = (
                symbol if dim.scale == 1 else f"{dim.scale}*{symbol}"
            )
            shape.append(
                jax.export.symbolic_shape(expression, scope=scope)[0]
            )
    return jax.ShapeDtypeStruct(  # type: ignore[no-untyped-call]
        tuple(shape),
        jnp.dtype(value.dtype),
        weak_type=value.weak_type,
    )


def _trace_key(
    ordered_values: tuple[Any, ...],
    abstract_values: tuple[Any, ...],
    static_indices: frozenset[int],
) -> tuple[Any, ...]:
    items: list[Any] = []
    for index, (value, abstract) in enumerate(
        zip(ordered_values, abstract_values, strict=True)
    ):
        if index in static_indices:
            items.append(("static", type(value), value))
            continue

        leaves, tree = jax.tree.flatten(abstract)
        leaf_keys: list[Any] = []
        for leaf in leaves:
            aval = jax.typeof(leaf)
            leaf_keys.append(
                (
                    tuple(str(dim) for dim in aval.shape),
                    str(aval.dtype),
                    bool(getattr(aval, "weak_type", False)),
                )
            )
        items.append(("dynamic", str(tree), tuple(leaf_keys)))
    return tuple(items)


def _normalize_input_leaf(value: Any) -> InputValue:
    if isinstance(value, (PrivArray, SensitiveArray, jax.Array)):
        return value
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    if isinstance(value, (bool, int, float)):
        return jnp.asarray(value)
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, (bool, int, float)):
            return jnp.asarray(scalar)
    raise TypeError(f"Unsupported call input leaf: {type(value).__name__}")


class JitWrapped(Generic[P, R]):
    def __init__(
        self,
        function: Callable[P, R],
        *,
        static_argnums: int | Sequence[int] = (),
        static_argnames: str | Iterable[str] = (),
    ) -> None:
        self._function = function
        self._signature = inspect.signature(function)
        self._parameters = tuple(self._signature.parameters.values())
        self._static_indices = _static_indices(
            self._signature,
            static_argnums,
            static_argnames,
        )
        self._cache: dict[tuple[Any, ...], _CachedComputation] = {}
        update_wrapper(self, function)

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        ordered_values = tuple(
            bound.arguments[parameter.name] for parameter in self._parameters
        )
        static_indices = frozenset(self._static_indices)
        for index in static_indices:
            _assert_static_value(ordered_values[index])

        scope = jax.export.SymbolicScope()
        alignment_symbols: dict[AlignmentSignature, str] = {}

        def abstract_leaf(leaf: Any) -> Any:
            if isinstance(leaf, (PrivArray, SensitiveArray)):
                return _symbolic_abstract(
                    leaf,
                    scope,
                    alignment_symbols,
                )
            return _public_abstract(leaf)

        abstract_values = tuple(
            value if index in static_indices else jax.tree.map(abstract_leaf, value)
            for index, value in enumerate(ordered_values)
        )
        dynamic_values = tuple(
            value
            for index, value in enumerate(ordered_values)
            if index not in static_indices
        )
        flat_inputs = [
            _normalize_input_leaf(leaf) for leaf in jax.tree.leaves(dynamic_values)
        ]
        key = _trace_key(ordered_values, abstract_values, static_indices)
        cached = self._cache.get(key)

        if cached is None:

            def ordered_function(*values: Any) -> Any:
                return _call_ordered(self._function, self._parameters, values)

            traced_result = jax.make_jaxpr(
                ordered_function,
                static_argnums=self._static_indices,
                return_shape=True,
            )(*abstract_values)
            closed_jaxpr, output_shape = traced_result
            batch = lower_jaxpr(closed_jaxpr, flat_inputs)
            cached = _CachedComputation(
                computation=jit_computation(batch),
                out_tree=jax.tree.structure(output_shape),
            )
            self._cache[key] = cached

        flat_results = cached.computation.call_flat(flat_inputs)
        return cast(R, cached.out_tree.unflatten(flat_results))


@overload
def jit(
    function: Callable[P, R],
    *,
    static_argnums: int | Sequence[int] = (),
    static_argnames: str | Iterable[str] = (),
) -> JitWrapped[P, R]: ...


@overload
def jit(
    function: None = None,
    *,
    static_argnums: int | Sequence[int] = (),
    static_argnames: str | Iterable[str] = (),
) -> Callable[[Callable[P, R]], JitWrapped[P, R]]: ...


def jit(
    function: Callable[P, R] | None = None,
    *,
    static_argnums: int | Sequence[int] = (),
    static_argnames: str | Iterable[str] = (),
) -> JitWrapped[P, R] | Callable[[Callable[P, R]], JitWrapped[P, R]]:
    def wrap(target: Callable[P, R]) -> JitWrapped[P, R]:
        return JitWrapped(
            target,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        )

    if function is None:
        return wrap
    return wrap(function)


__all__ = ["JitWrapped", "jit"]
