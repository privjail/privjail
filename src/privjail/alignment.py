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

AxisSignature = int

_axis_signature_counter: AxisSignature = 0

def new_axis_signature() -> AxisSignature:
    global _axis_signature_counter
    _axis_signature_counter += 1
    return _axis_signature_counter

@runtime_checkable
class AxisAligned(Protocol):
    _distance_axis  : int
    _axis_signature : AxisSignature

def assert_axis_signature(*arrays: AxisAligned) -> None:
    if len(arrays) > 0 and not all(arrays[0]._axis_signature == arr._axis_signature for arr in arrays):
        raise DPError("Axis signatures do not match")

def assert_distance_axis(*arrays: AxisAligned) -> None:
    if len(arrays) > 0 and not all(arrays[0]._distance_axis == arr._distance_axis for arr in arrays):
        raise DPError("Distance axes do not match")
    assert_axis_signature(*arrays)
