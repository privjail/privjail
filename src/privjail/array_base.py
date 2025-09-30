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

from .alignment import AxisAligned, AxisSignature, assert_axis_signature, new_axis_signature
from .accountants import Accountant
from .prisoner import Prisoner
from .realexpr import RealExpr

T = TypeVar("T")

class PrivArrayBase(Generic[T], Prisoner[T], AxisAligned):
    _axis_signature: AxisSignature

    def __init__(self,
                 value        : Any,
                 distance     : RealExpr,
                 parents      : Sequence[PrivArrayBase[Any]],
                 accountant   : Accountant[Any] | None,
                 preserve_row : bool | None,
                 ) -> None:
        if len(parents) == 0:
            preserve_row = False
        elif preserve_row is None:
            raise ValueError("preserve_row is required when parents are specified.")

        if preserve_row:
            assert_axis_signature(*parents)
            self._axis_signature = parents[0]._axis_signature
        else:
            self._axis_signature = new_axis_signature()

        super().__init__(value=value, distance=distance, parents=parents, accountant=accountant)

    def renew_axis_signature(self) -> None:
        self._axis_signature = new_axis_signature()
