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
from .alignment import AxisAligned, AlignmentSignature, assert_privacy_axis, new_alignment_signature
from .accountants import Accountant
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat
from .realexpr import RealExpr
from .util import integer, floating

T = TypeVar("T")

class PrivArrayBase(Generic[T], Prisoner[T], AxisAligned):
    _privacy_axis        : int
    _alignment_signature : AlignmentSignature

    def __init__(self,
                 value          : Any,
                 distance       : RealExpr,
                 privacy_axis   : int,
                 parents        : Sequence[PrivArrayBase[Any]],
                 accountant     : Accountant[Any] | None,
                 *,
                 keep_alignment : bool,
                 ) -> None:
        if privacy_axis < 0:
            raise ValueError("privacy_axis must be non-negative.")

        if len(parents) == 0 and keep_alignment:
            raise ValueError("keep_alignment=True requires at least one parent.")

        self._privacy_axis = privacy_axis

        if keep_alignment:
            assert_privacy_axis(*parents)
            self._alignment_signature = parents[0]._alignment_signature
        else:
            self._alignment_signature = new_alignment_signature()

        super().__init__(value=value, distance=distance, parents=parents, accountant=accountant)

    @egrpc.property
    def privacy_axis(self) -> int:
        return self._privacy_axis

    @egrpc.property
    def alignment_signature(self) -> int:
        return self._alignment_signature

    def renew_alignment_signature(self) -> None:
        self._alignment_signature = new_alignment_signature()

@egrpc.remoteclass
class SensitiveDimInt(SensitiveInt):
    _alignment_signature : AlignmentSignature
    _scale               : int

    def __init__(self,
                 value               : integer,
                 distance            : RealExpr,
                 alignment_signature : AlignmentSignature,
                 scale               : int                     = 1,
                 *,
                 parents             : Sequence[Prisoner[Any]] = [],
                 accountant          : Accountant[Any] | None  = None,
                 ) -> None:
        self._alignment_signature = alignment_signature
        self._scale = scale
        super().__init__(value, distance, parents=parents, accountant=accountant)

    @egrpc.property
    def alignment_signature(self) -> AlignmentSignature:
        return self._alignment_signature

    @egrpc.property
    def scale(self) -> int:
        return self._scale

    @egrpc.method
    def __neg__(self) -> SensitiveDimInt:
        return SensitiveDimInt(-self._value,
                               distance            = self._distance,
                               alignment_signature = self._alignment_signature,
                               scale               = -self._scale,
                               parents             = [self])

    @egrpc.multimethod
    def __add__(self, other: SensitiveDimInt) -> SensitiveDimInt | SensitiveInt:
        if self._alignment_signature == other._alignment_signature:
            return SensitiveDimInt(self._value + other._value,
                                   distance            = self._distance + other._distance,
                                   alignment_signature = self._alignment_signature,
                                   scale               = self._scale + other._scale,
                                   parents             = [self, other])
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
        if self._alignment_signature == other._alignment_signature:
            return SensitiveDimInt(self._value - other._value,
                                   distance            = self._distance + other._distance,
                                   alignment_signature = self._alignment_signature,
                                   scale               = self._scale - other._scale,
                                   parents             = [self, other])
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
                               distance            = self._distance * abs(int(other)),
                               alignment_signature = self._alignment_signature,
                               scale               = self._scale * int(other),
                               parents             = [self])

    @__mul__.register
    def _(self, other: floating) -> SensitiveFloat:
        return super().__mul__(other)  # type: ignore[no-any-return]

    @egrpc.multimethod
    def __rmul__(self, other: integer) -> SensitiveDimInt:  # type: ignore[misc]
        return SensitiveDimInt(other * self._value,
                               distance            = self._distance * abs(int(other)),
                               alignment_signature = self._alignment_signature,
                               scale               = int(other) * self._scale,
                               parents             = [self])

    @__rmul__.register
    def _(self, other: floating) -> SensitiveFloat:
        return super().__rmul__(other)  # type: ignore[no-any-return]
