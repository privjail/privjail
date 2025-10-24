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
from typing import Any, Generic, Sequence, TypeVar

from . import egrpc
from .alignment import AxisAligned, AxisSignature, assert_distance_axis, new_axis_signature
from .accountants import Accountant
from .prisoner import Prisoner
from .realexpr import RealExpr

T = TypeVar("T")

class PrivArrayBase(Generic[T], Prisoner[T], AxisAligned):
    _distance_axis  : int
    _axis_signature : AxisSignature

    def __init__(self,
                 value                  : Any,
                 distance               : RealExpr,
                 distance_axis          : int,
                 parents                : Sequence[PrivArrayBase[Any]],
                 accountant             : Accountant[Any] | None,
                 *,
                 inherit_axis_signature : bool,
                 ) -> None:
        if distance_axis < 0:
            raise ValueError("distance_axis must be non-negative.")

        if len(parents) == 0 and inherit_axis_signature:
            raise ValueError("inherit_axis_signature=True requires at least one parent.")

        self._distance_axis = distance_axis

        if inherit_axis_signature:
            assert_distance_axis(*parents)
            self._axis_signature = parents[0]._axis_signature
        else:
            self._axis_signature = new_axis_signature()

        super().__init__(value=value, distance=distance, parents=parents, accountant=accountant)

    @egrpc.property
    def distance_axis(self) -> int:
        return self._distance_axis

    @egrpc.property
    def axis_signature(self) -> int:
        return self._axis_signature

    def renew_axis_signature(self) -> None:
        self._axis_signature = new_axis_signature()
