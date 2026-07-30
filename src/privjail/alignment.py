# Copyright 2025 TOYOTA MOTOR CORPORATION.
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
from typing import Protocol, runtime_checkable

import egrpc

from .realexpr import RealExpr
from .util import DPError

@egrpc.dataclass
class AlignmentSignature:
    base  : int
    left  : int = 1
    right : int = 1

    def __post_init__(self) -> None:
        if self.base <= 0 or self.left <= 0 or self.right <= 0:
            raise ValueError("Alignment factors must be positive.")

    def __hash__(self) -> int:
        return hash((self.base, self.left, self.right))

_alignment_signature_counter = 0

def new_alignment_signature() -> AlignmentSignature:
    global _alignment_signature_counter
    _alignment_signature_counter += 1
    return AlignmentSignature(_alignment_signature_counter)

def alignment_factor(signature: AlignmentSignature) -> int:
    return signature.left * signature.right

def reshape_alignment_signature(
    input_signature           : AlignmentSignature,
    dimension_signature       : AlignmentSignature,
    dimension_scale           : int,
    *,
    input_prefix              : int,
    input_suffix              : int,
    output_prefix             : int,
    output_suffix             : int,
) -> AlignmentSignature:
    # C-order layout is prefix · left · base · right · suffix.
    if input_signature.base != dimension_signature.base:
        raise DPError("A private reshape dimension must share its input base.")
    if (
        dimension_scale <= 0
        or input_prefix <= 0
        or input_suffix <= 0
        or output_prefix <= 0
        or output_suffix <= 0
    ):
        raise DPError("Private reshape layout factors must be positive.")

    left_numerator = input_prefix * input_signature.left
    right_numerator = input_signature.right * input_suffix
    if (
        left_numerator % output_prefix != 0
        or right_numerator % output_suffix != 0
    ):
        raise DPError("Reshape would mix data across individuals.")

    result = AlignmentSignature(
        base=input_signature.base,
        left=left_numerator // output_prefix,
        right=right_numerator // output_suffix,
    )
    if alignment_factor(result) != (
        alignment_factor(dimension_signature) * dimension_scale
    ):
        raise DPError("Reshape would mix data across individuals.")
    return result

def assert_normalized_distance(
    first_distance  : RealExpr,
    first_alignment : AlignmentSignature,
    second_distance : RealExpr,
    second_alignment: AlignmentSignature,
) -> None:
    # distance / (left * right) is invariant under an exact layout change.
    if (
        first_alignment.base != second_alignment.base
        or not (
            first_distance * alignment_factor(second_alignment)
        ).structurally_equal(
            second_distance * alignment_factor(first_alignment)
        )
    ):
        raise DPError("Distances do not match the alignment layout.")

@runtime_checkable
class AxisAligned(Protocol):
    _privacy_axis        : int
    _alignment_signature : AlignmentSignature

def assert_alignment_signature(*arrays: AxisAligned) -> None:
    if len(arrays) > 0 and not all(arrays[0]._alignment_signature == arr._alignment_signature for arr in arrays):
        raise DPError("Alignment signatures do not match")

def assert_privacy_axis(*arrays: AxisAligned) -> None:
    if len(arrays) > 0 and not all(arrays[0]._privacy_axis == arr._privacy_axis for arr in arrays):
        raise DPError("Privacy axes do not match")
    assert_alignment_signature(*arrays)
