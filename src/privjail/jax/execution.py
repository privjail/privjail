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
from dataclasses import dataclass
from typing import Any, Callable

import egrpc
import jax

from ..accountants import Accountant, RDPAccountant, RDPSubsamplingAccountant
from ..accountants.util import accounting_trace
from ..alignment import AlignmentSignature, new_alignment_signature
from ..realexpr import RealExpr, bind_l2_variables
from ..util import DPError
from .array import PrivArray, SensitiveArray
from .call_batch import (
    InputValue,
    ResultValue,
    validate_call_batch,
)
from .domain import ArrayDomain
from .primitives import PrimitiveValue, as_jax_value


OutputTemplate = PrivArray | SensitiveArray | None
Executable = Callable[..., tuple[jax.Array, ...]]


def _accountant_trace_key(
    accountant: Accountant[Any],
) -> tuple[Any, ...]:
    if not isinstance(accountant, RDPAccountant):
        return (type(accountant),)
    parent = accountant.parent
    sampling_rate = (
        parent.sampling_rate
        if isinstance(parent, RDPSubsamplingAccountant)
        else None
    )
    return (
        type(accountant),
        tuple(accountant.alpha),
        sampling_rate,
    )


@dataclass(frozen=True)
class _OutputAlignment:
    base_slot: int
    left: int
    right: int


@dataclass(frozen=True)
class _SemanticReplay:
    templates: tuple[OutputTemplate, ...]
    output_alignments: tuple[_OutputAlignment | None, ...]
    input_l2_variables: tuple[Any, ...]
    accounting_budgets: tuple[Any, ...]
    executables: dict[tuple[Any, ...], Executable]


def _raw_value(value: InputValue) -> Any:
    if isinstance(value, (PrivArray, SensitiveArray)):
        return value._value
    return value


def _concrete_key(inputs: list[InputValue]) -> tuple[Any, ...]:
    key: list[Any] = []
    for value in inputs:
        aval = jax.typeof(_raw_value(value))
        key.append(
            (
                tuple(int(dim) for dim in aval.shape),
                str(aval.dtype),
                bool(getattr(aval, "weak_type", False)),
            )
        )
    return tuple(key)


def _clone_input(
    value: InputValue,
    raw_value: jax.Array,
) -> PrimitiveValue:
    if isinstance(value, PrivArray):
        return PrivArray(
            value=raw_value,
            distance=value._distance,
            privacy_axis=value._privacy_axis,
            domain=value._domain,
            parents=[value],
            keep_alignment=True,
        )

    if isinstance(value, SensitiveArray):
        return SensitiveArray(
            value=raw_value,
            distance=value._distance,
            norm_type=value._norm_type,
            parents=[value],
        )

    return raw_value


def _metadata_expressions(inputs: list[InputValue]) -> list[RealExpr]:
    expressions: list[RealExpr] = []
    for value in inputs:
        if isinstance(value, PrivArray):
            expressions.append(value._distance)
            if value._domain.norm_bound is not None:
                expressions.append(value._domain.norm_bound)
        elif isinstance(value, SensitiveArray):
            expressions.append(value._distance)
    return expressions


def _canonical_l2_variables(inputs: list[InputValue]) -> tuple[Any, ...]:
    """Order fresh L2 variables by their semantic input positions.

    A joint constraint appears on every component expression. The first pass
    assigns the component used by each expression before the second pass adds
    constraint-only siblings, so fresh SymPy names do not affect cache keys.
    """
    l2_variable_slots: dict[Any, int] = {}
    expressions = _metadata_expressions(inputs)
    for expression in expressions:
        expression_variables: set[Any] | frozenset[Any] = getattr(
            expression.expr,
            "free_symbols",
            frozenset(),
        )
        for variable in sorted(
            expression.l2_variables() & expression_variables,
            key=str,
        ):
            l2_variable_slots.setdefault(variable, len(l2_variable_slots))
    for expression in expressions:
        for variable in sorted(expression.l2_variables(), key=str):
            l2_variable_slots.setdefault(variable, len(l2_variable_slots))
    return tuple(l2_variable_slots)


def _semantic_key(
    inputs: list[InputValue],
    l2_variables: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Describe all input metadata that can change trusted replay semantics.

    Only the concrete privacy-axis size is omitted. Each new concrete shape is
    replayed before compilation and must produce the same accounting budgets;
    all other shapes, alignment relations, accountant parameters, distances,
    and domains participate in this key.
    """
    key: list[Any] = []
    alignment_slots: dict[int, int] = {}
    l2_variable_slots = {
        variable: slot for slot, variable in enumerate(l2_variables)
    }

    for value in inputs:
        raw = _raw_value(value)
        aval = jax.typeof(raw)
        if isinstance(value, PrivArray):
            alignment_slot = alignment_slots.setdefault(
                value._alignment_signature.base,
                len(alignment_slots),
            )
            key.append(
                (
                    "private",
                    tuple(
                        "*" if axis == value._privacy_axis else int(dim)
                        for axis, dim in enumerate(aval.shape)
                    ),
                    str(aval.dtype),
                    value._privacy_axis,
                    (
                        alignment_slot,
                        value._alignment_signature.left,
                        value._alignment_signature.right,
                    ),
                    _accountant_trace_key(value._accountant),
                    value._distance.semantic_key(l2_variable_slots),
                    (
                        None
                        if value._domain.norm_bound is None
                        else value._domain.norm_bound.semantic_key(
                            l2_variable_slots
                        )
                    ),
                    value._domain.norm_type,
                    value._domain.value_range,
                )
            )
        elif isinstance(value, SensitiveArray):
            key.append(
                (
                    "sensitive",
                    tuple(int(dim) for dim in aval.shape),
                    str(aval.dtype),
                    _accountant_trace_key(value._accountant),
                    value._distance.semantic_key(l2_variable_slots),
                    value._norm_type,
                )
            )
        else:
            key.append(
                (
                    "public",
                    tuple(int(dim) for dim in aval.shape),
                    str(aval.dtype),
                    bool(getattr(aval, "weak_type", False)),
                )
            )
    return tuple(key)


def _input_alignment_bases(
    inputs: Sequence[Any],
) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            value._alignment_signature.base
            for value in inputs
            if isinstance(value, PrivArray)
        )
    )


def _output_alignments(
    inputs: list[PrimitiveValue],
    outputs: list[PrimitiveValue],
) -> tuple[_OutputAlignment | None, ...]:
    slots = {
        base: slot
        for slot, base in enumerate(
            _input_alignment_bases(inputs)
        )
    }
    result: list[_OutputAlignment | None] = []
    for output in outputs:
        if not isinstance(output, PrivArray):
            result.append(None)
            continue
        signature = output._alignment_signature
        result.append(
            _OutputAlignment(
                base_slot=slots.setdefault(
                    signature.base,
                    len(slots),
                ),
                left=signature.left,
                right=signature.right,
            )
        )
    return tuple(result)


def _bind_output_alignments(
    replay: _SemanticReplay,
    inputs: list[InputValue],
) -> tuple[AlignmentSignature | None, ...]:
    bases = list(_input_alignment_bases(inputs))
    output_slots = [
        alignment.base_slot
        for alignment in replay.output_alignments
        if alignment is not None
    ]
    required = 0 if not output_slots else max(output_slots) + 1
    bases.extend(
        new_alignment_signature().base
        for _ in range(required - len(bases))
    )
    return tuple(
        (
            None
            if alignment is None
            else AlignmentSignature(
                base=bases[alignment.base_slot],
                left=alignment.left,
                right=alignment.right,
            )
        )
        for alignment in replay.output_alignments
    )


def _call_accountant(
    inputs: list[InputValue],
) -> Accountant[Any] | None:
    accountants = [
        value._accountant
        for value in inputs
        if isinstance(value, (PrivArray, SensitiveArray))
    ]
    if not accountants:
        return None
    accountant = accountants[0]
    if any(candidate is not accountant for candidate in accountants[1:]):
        raise DPError(
            "All protected inputs to one traced computation must share "
            "one accountant."
        )
    return accountant


def _materialize(
    template: OutputTemplate,
    value: jax.Array,
    alignment_signature: AlignmentSignature | None,
    accountant: Accountant[Any] | None,
    l2_variable_mapping: dict[Any, Any],
) -> ResultValue:
    if isinstance(template, PrivArray):
        if accountant is None:
            raise RuntimeError(
                "A private output requires a call accountant."
            )
        norm_bound = template._domain.norm_bound
        result = PrivArray(
            value=value,
            distance=bind_l2_variables(
                template._distance,
                l2_variable_mapping,
            ),
            privacy_axis=template._privacy_axis,
            domain=ArrayDomain(
                norm_type=template._domain.norm_type,
                norm_bound=(
                    None
                    if norm_bound is None
                    else bind_l2_variables(
                        norm_bound,
                        l2_variable_mapping,
                    )
                ),
                value_range=template._domain.value_range,
            ),
            accountant=accountant,
        )
        if alignment_signature is None:
            raise RuntimeError("A private output requires an alignment signature.")
        result._alignment_signature = alignment_signature
        return result

    if isinstance(template, SensitiveArray):
        if alignment_signature is not None:
            raise RuntimeError("A sensitive output cannot have an alignment.")
        if accountant is None:
            raise RuntimeError(
                "A sensitive output requires a call accountant."
            )
        return SensitiveArray(
            value=value,
            distance=bind_l2_variables(
                template._distance,
                l2_variable_mapping,
            ),
            norm_type=template._norm_type,
            accountant=accountant,
        )

    if alignment_signature is not None:
        raise RuntimeError("A public output cannot have an alignment.")
    return value


@egrpc.remoteclass
class JitComputation:
    def __init__(self, batch: egrpc.CallBatch):
        self._input_count = validate_call_batch(batch)
        self._batch = batch
        self._semantic_replays: dict[tuple[Any, ...], _SemanticReplay] = {}

    def _replay_semantics(
        self,
        inputs: list[InputValue],
        input_l2_variables: tuple[Any, ...],
    ) -> _SemanticReplay:
        traced_outputs: tuple[PrimitiveValue, ...] | None = None
        output_alignments: tuple[_OutputAlignment | None, ...] | None = None

        def replay(*raw_inputs: Any) -> tuple[jax.Array, ...]:
            nonlocal traced_outputs
            nonlocal output_alignments

            accounting_budgets.clear()
            trace_inputs = [
                _clone_input(value, raw)
                for value, raw in zip(inputs, raw_inputs, strict=True)
            ]
            outputs = self._batch(*trace_inputs)
            if not isinstance(outputs, list):
                raise RuntimeError("pack_outputs did not return a list.")
            traced_outputs = tuple(outputs)
            output_alignments = _output_alignments(
                trace_inputs,
                outputs,
            )
            return tuple(as_jax_value(output) for output in outputs)

        raw_inputs = tuple(_raw_value(value) for value in inputs)
        concrete_key = _concrete_key(inputs)
        with accounting_trace() as accounting_budgets:
            executable = jax.jit(replay).lower(
                *raw_inputs,
            ).compile()
        if (
            traced_outputs is None
            or output_alignments is None
        ):
            raise RuntimeError("Trusted replay did not produce output templates.")
        templates: list[OutputTemplate] = []
        for output in traced_outputs:
            if isinstance(output, (PrivArray, SensitiveArray)):
                output._value = None  # type: ignore[assignment]
                templates.append(output)
            else:
                templates.append(None)
        return _SemanticReplay(
            templates=tuple(templates),
            output_alignments=output_alignments,
            input_l2_variables=input_l2_variables,
            accounting_budgets=tuple(accounting_budgets),
            executables={concrete_key: executable},
        )

    @egrpc.method
    def call_flat(self, inputs: list[InputValue]) -> list[ResultValue]:
        if len(inputs) != self._input_count:
            raise TypeError(
                f"Expected {self._input_count} inputs, got {len(inputs)}."
            )

        accountant = _call_accountant(inputs)
        input_l2_variables = _canonical_l2_variables(inputs)
        key = _semantic_key(inputs, input_l2_variables)
        replay = self._semantic_replays.get(key)
        if replay is None:
            replay = self._replay_semantics(inputs, input_l2_variables)
            self._semantic_replays[key] = replay

        concrete_key = _concrete_key(inputs)
        executable = replay.executables.get(concrete_key)
        if executable is None:
            concrete_replay = self._replay_semantics(
                inputs,
                input_l2_variables,
            )
            if (
                concrete_replay.accounting_budgets
                != replay.accounting_budgets
            ):
                raise RuntimeError(
                    "Accounting semantics depend on a concrete private shape."
                )
            executable = concrete_replay.executables[concrete_key]
            replay.executables[concrete_key] = executable

        if replay.accounting_budgets and accountant is None:
            raise DPError(
                "A private accountant is required for a DP release."
            )
        if accountant is not None:
            for budget in replay.accounting_budgets:
                accountant.spend(budget)
        raw_results = executable(
            *tuple(_raw_value(value) for value in inputs),
        )
        l2_variable_mapping = dict(
            zip(
                replay.input_l2_variables,
                input_l2_variables,
                strict=True,
            )
        )
        alignment_signatures = _bind_output_alignments(replay, inputs)
        return [
            _materialize(
                template,
                raw_result,
                alignment_signature,
                accountant,
                l2_variable_mapping,
            )
            for template, raw_result, alignment_signature in zip(
                replay.templates,
                raw_results,
                alignment_signatures,
                strict=True,
            )
        ]


@egrpc.function
def jit_computation(batch: egrpc.CallBatch) -> JitComputation:
    return JitComputation(batch)


__all__ = [
    "InputValue",
    "ResultValue",
    "JitComputation",
    "jit_computation",
]
