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

from .util import DPError

AlignmentSignature = int

_alignment_signature_counter: AlignmentSignature = 0

def new_alignment_signature() -> AlignmentSignature:
    global _alignment_signature_counter
    _alignment_signature_counter += 1
    return _alignment_signature_counter

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
