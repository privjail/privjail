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
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat
from .realexpr import RealExpr
from .util import integer, floating

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

@egrpc.remoteclass
class SensitiveDimInt(SensitiveInt):
    _axis_signature : AxisSignature
    _scale          : int

    def __init__(self,
                 value          : integer,
                 distance       : RealExpr,
                 axis_signature : AxisSignature,
                 scale          : int                      = 1,
                 *,
                 parents        : Sequence[Prisoner[Any]]  = [],
                 accountant     : Accountant[Any] | None   = None,
                 ) -> None:
        self._axis_signature = axis_signature
        self._scale = scale
        super().__init__(value, distance, parents=parents, accountant=accountant)

    @egrpc.property
    def axis_signature(self) -> AxisSignature:
        return self._axis_signature

    @egrpc.property
    def scale(self) -> int:
        return self._scale

    @egrpc.method
    def __neg__(self) -> SensitiveDimInt:
        return SensitiveDimInt(-self._value,
                               distance       = self.distance,
                               axis_signature = self._axis_signature,
                               scale          = -self._scale,
                               parents        = [self])

    @egrpc.multimethod
    def __add__(self, other: SensitiveDimInt) -> SensitiveDimInt | SensitiveInt:
        if self._axis_signature == other._axis_signature:
            return SensitiveDimInt(self._value + other._value,
                                   distance       = self.distance + other.distance,
                                   axis_signature = self._axis_signature,
                                   scale          = self._scale + other._scale,
                                   parents        = [self, other])
        return super().__add__(other)  # type: ignore[no-any-return]

    @__add__.register
    def _(self, other: integer) -> SensitiveInt:
        return super().__add__(other)  # type: ignore[no-any-return]

    @__add__.register
    def _(self, other: floating) -> SensitiveFloat:
        return super().__add__(other)  # type: ignore[no-any-return]

    @__add__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return super().__add__(other)  # type: ignore[no-any-return]

    @__add__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return super().__add__(other)  # type: ignore[no-any-return]

    @egrpc.multimethod
    def __sub__(self, other: SensitiveDimInt) -> SensitiveDimInt | SensitiveInt:
        if self._axis_signature == other._axis_signature:
            return SensitiveDimInt(self._value - other._value,
                                   distance       = self.distance + other.distance,
                                   axis_signature = self._axis_signature,
                                   scale          = self._scale - other._scale,
                                   parents        = [self, other])
        return super().__sub__(other)  # type: ignore[no-any-return]

    @__sub__.register
    def _(self, other: integer) -> SensitiveInt:
        return super().__sub__(other)  # type: ignore[no-any-return]

    @__sub__.register
    def _(self, other: floating) -> SensitiveFloat:
        return super().__sub__(other)  # type: ignore[no-any-return]

    @__sub__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return super().__sub__(other)  # type: ignore[no-any-return]

    @__sub__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return super().__sub__(other)  # type: ignore[no-any-return]

    @egrpc.multimethod
    def __mul__(self, other: integer) -> SensitiveDimInt:
        return SensitiveDimInt(self._value * other,
                               distance       = self.distance * abs(int(other)),
                               axis_signature = self._axis_signature,
                               scale          = self._scale * int(other),
                               parents        = [self])

    @__mul__.register
    def _(self, other: floating) -> SensitiveFloat:
        return super().__mul__(other)  # type: ignore[no-any-return]

    @egrpc.multimethod
    def __rmul__(self, other: integer) -> SensitiveDimInt:  # type: ignore[misc]
        return SensitiveDimInt(other * self._value,
                               distance       = self.distance * abs(int(other)),
                               axis_signature = self._axis_signature,
                               scale          = int(other) * self._scale,
                               parents        = [self])

    @__rmul__.register
    def _(self, other: floating) -> SensitiveFloat:
        return super().__rmul__(other)  # type: ignore[no-any-return]
