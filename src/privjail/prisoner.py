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
from typing import TypeVar, Generic, Any, overload, Iterable, cast, Sequence

import numpy as _np

from .util import integer, floating, realnum, is_integer, is_floating
from .provenance import ProvenanceEntity, new_provenance_root, new_provenance_node, consume_privacy_budget, consumed_privacy_budget_all, ChildrenType
from .distance import Distance, _max as dmax
from . import egrpc

T = TypeVar("T")

class Prisoner(Generic[T]):
    _value     : T
    distance   : Distance
    provenance : list[ProvenanceEntity]

    def __init__(self,
                 value         : T,
                 distance      : Distance,
                 *,
                 parents       : Sequence[Prisoner[Any]] = [],
                 root_name     : str | None              = None,
                 children_type : ChildrenType            = "inclusive",
                 ):
        self._value   = value
        self.distance = distance

        if distance.is_zero():
            self.provenance = []

        elif len(parents) == 0:
            if root_name is None:
                raise ValueError("Both parents and root_name are not specified.")

            self.provenance = [new_provenance_root(root_name)]

        elif children_type == "inclusive":
            parent_provenance = list({pe for p in parents for pe in p.provenance})

            if len(parents) == 1 and parents[0].provenance[0].children_type == "exclusive":
                self.provenance = [new_provenance_node(parent_provenance, "inclusive")]
            else:
                self.provenance = parent_provenance

        elif children_type == "exclusive":
            parent_provenance = list({pe for p in parents for pe in p.provenance})
            self.provenance = [new_provenance_node(parent_provenance, "exclusive")]

        else:
            raise RuntimeError

    def __str__(self) -> str:
        # return f"Prisoner({type(self._value)}, distance={self.distance.max()})"
        return f"Prisoner()"

    def __repr__(self) -> str:
        # return f"Prisoner({type(self._value)}, distance={self.distance.max()})"
        return f"Prisoner()"

    def consume_privacy_budget(self, privacy_budget: float) -> None:
        consume_privacy_budget(self.provenance, privacy_budget)

    def root_name(self) -> str:
        assert len(self.provenance) > 0
        return self.provenance[0].root_name

@egrpc.remoteclass
class SensitiveInt(Prisoner[integer]):
    def __init__(self,
                 value         : integer,
                 distance      : Distance                = Distance(0),
                 *,
                 parents       : Sequence[Prisoner[Any]] = [],
                 root_name     : str | None              = None,
                 children_type : ChildrenType            = "inclusive",
                 ):
        if not is_integer(value):
            raise ValueError("`value` must be int for SensitveInt.")
        super().__init__(value, distance, parents=parents, root_name=root_name, children_type=children_type)

    @egrpc.multimethod
    def __add__(self, other: integer) -> SensitiveInt:
        return SensitiveInt(self._value + other, distance=self.distance, parents=[self])

    @__add__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(self._value + other, distance=self.distance, parents=[self])

    @__add__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return SensitiveInt(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])

    @__add__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __radd__(self, other: integer) -> SensitiveInt: # type: ignore[misc]
        return SensitiveInt(other + self._value, distance=self.distance, parents=[self])

    @__radd__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(other + self._value, distance=self.distance, parents=[self])

    @__radd__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return SensitiveInt(other._value + self._value, distance=self.distance + other.distance, parents=[self, other])

    @__radd__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(other._value + self._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __sub__(self, other: integer) -> SensitiveInt:
        return SensitiveInt(self._value - other, distance=self.distance, parents=[self])

    @__sub__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(self._value - other, distance=self.distance, parents=[self])

    @__sub__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return SensitiveInt(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])

    @__sub__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __rsub__(self, other: integer) -> SensitiveInt: # type: ignore[misc]
        return SensitiveInt(other - self._value, distance=self.distance, parents=[self])

    @__rsub__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(other - self._value, distance=self.distance, parents=[self])

    @__rsub__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return SensitiveInt(other._value - self._value, distance=self.distance + other.distance, parents=[self, other])

    @__rsub__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(other._value - self._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __mul__(self, other: integer) -> SensitiveInt:
        return SensitiveInt(self._value * other, distance=self.distance * _np.abs(other), parents=[self])

    @__mul__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(self._value * other, distance=self.distance * _np.abs(other), parents=[self])

    @egrpc.multimethod
    def __rmul__(self, other: integer) -> SensitiveInt: # type: ignore[misc]
        return SensitiveInt(other * self._value, distance=self.distance * _np.abs(other), parents=[self])

    @__rmul__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(other * self._value, distance=self.distance * _np.abs(other), parents=[self])

@egrpc.remoteclass
class SensitiveFloat(Prisoner[floating]):
    def __init__(self,
                 value         : floating,
                 distance      : Distance                = Distance(0),
                 *,
                 parents       : Sequence[Prisoner[Any]] = [],
                 root_name     : str | None              = None,
                 children_type : ChildrenType            = "inclusive",
                 ):
        if not is_floating(value):
            raise ValueError("`value` must be float for SensitveFloat.")
        super().__init__(value, distance, parents=parents, root_name=root_name, children_type=children_type)

    @egrpc.multimethod
    def __add__(self, other: realnum) -> SensitiveFloat:
        return SensitiveFloat(self._value + other, distance=self.distance, parents=[self])

    @__add__.register
    def _(self, other: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __radd__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        return SensitiveFloat(other + self._value, distance=self.distance, parents=[self])

    @__radd__.register
    def _(self, other: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(other._value + self._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __sub__(self, other: realnum) -> SensitiveFloat:
        return SensitiveFloat(self._value - other, distance=self.distance, parents=[self])

    @__sub__.register
    def _(self, other: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __rsub__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        return SensitiveFloat(other - self._value, distance=self.distance, parents=[self])

    @__rsub__.register
    def _(self, other: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(other._value - self._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __mul__(self, other: realnum) -> SensitiveFloat:
        return SensitiveFloat(self._value * other, distance=self.distance * _np.abs(other), parents=[self])

    @egrpc.multimethod
    def __rmul__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        return SensitiveFloat(other * self._value, distance=self.distance * _np.abs(other), parents=[self])

@egrpc.multifunction
def max2(a: SensitiveInt | SensitiveFloat, b: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
    return SensitiveFloat(max(float(a._value), float(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

@max2.register
def _(a: SensitiveInt, b: SensitiveInt) -> SensitiveInt:
    return SensitiveInt(max(int(a._value), int(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

@overload
def _max(*args: SensitiveInt) -> SensitiveInt: ...
@overload
def _max(*args: SensitiveFloat) -> SensitiveFloat: ...
@overload
def _max(*args: Iterable[SensitiveInt]) -> SensitiveInt: ...
@overload
def _max(*args: Iterable[SensitiveFloat]) -> SensitiveFloat: ...
@overload
def _max(*args: Iterable[SensitiveInt | SensitiveFloat] | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat: ...

def _max(*args: Iterable[SensitiveInt | SensitiveFloat] | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat:
    if len(args) == 0:
        raise TypeError("max() expected at least one argment.")

    if len(args) == 1:
        if not isinstance(args[0], Iterable):
            raise TypeError("The first arg passed to max() is not iterable.")
        iterable = args[0]
    else:
        iterable = cast(tuple[SensitiveInt | SensitiveFloat, ...], args)

    it = iter(iterable)
    try:
        result = next(it)
    except StopIteration:
        raise ValueError("List passed to max() is empty.")

    for x in it:
        result = max2(result, x)

    return result

@egrpc.multifunction
def min2(a: SensitiveInt | SensitiveFloat, b: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
    return SensitiveFloat(min(float(a._value), float(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

@min2.register
def _(a: SensitiveInt, b: SensitiveInt) -> SensitiveInt:
    return SensitiveInt(min(int(a._value), int(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

@overload
def _min(*args: SensitiveInt) -> SensitiveInt: ...
@overload
def _min(*args: SensitiveFloat) -> SensitiveFloat: ...
@overload
def _min(*args: Iterable[SensitiveInt]) -> SensitiveInt: ...
@overload
def _min(*args: Iterable[SensitiveFloat]) -> SensitiveFloat: ...
@overload
def _min(*args: Iterable[SensitiveInt | SensitiveFloat] | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat: ...

def _min(*args: Iterable[SensitiveInt | SensitiveFloat] | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat:
    if len(args) == 0:
        raise TypeError("min() expected at least one argment.")

    if len(args) == 1:
        if not isinstance(args[0], Iterable):
            raise TypeError("The first arg passed to min() is not iterable.")
        iterable = args[0]
    else:
        iterable = cast(tuple[SensitiveInt | SensitiveFloat, ...], args)

    it = iter(iterable)
    try:
        result = next(it)
    except StopIteration:
        raise ValueError("List passed to min() is empty.")

    for x in it:
        result = min2(result, x)

    return result

@egrpc.function
def consumed_privacy_budget() -> dict[str, float]:
    return consumed_privacy_budget_all()
