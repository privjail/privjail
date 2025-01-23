from __future__ import annotations
from typing import TypeVar, Generic, Any, overload, Iterable, cast, Sequence

import numpy as _np

from .util import DPError, integer, floating, realnum, is_integer, is_floating, is_realnum
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
        return f"Prisoner({type(self._value)}, distance={self.distance.max()})"

    def __repr__(self) -> str:
        return f"Prisoner({type(self._value)}, distance={self.distance.max()})"

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

    @overload
    def __radd__(self, other: integer) -> SensitiveInt: ... # type: ignore[misc]
    @overload
    def __radd__(self, other: floating) -> SensitiveFloat: ... # type: ignore[misc]

    def __radd__(self, other: realnum) -> SensitiveInt | SensitiveFloat: # type: ignore[misc]
        if is_integer(other):
            return SensitiveInt(other + self._value, distance=self.distance, parents=[self])
        elif is_floating(other):
            return SensitiveFloat(other + self._value, distance=self.distance, parents=[self])
        else:
            raise ValueError("`other` must be a real number, SensitiveInt, or SensitiveFloat.")

    @overload
    def __sub__(self, other: integer | SensitiveInt) -> SensitiveInt: ...
    @overload
    def __sub__(self, other: floating | SensitiveFloat) -> SensitiveFloat: ...

    def __sub__(self, other: realnum | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat:
        if is_integer(other):
            return SensitiveInt(self._value - other, distance=self.distance, parents=[self])
        elif is_floating(other):
            return SensitiveFloat(self._value - other, distance=self.distance, parents=[self])
        elif isinstance(other, SensitiveInt):
            return SensitiveInt(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])
        elif isinstance(other, SensitiveFloat):
            return SensitiveFloat(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])
        else:
            raise ValueError("`other` must be a real number, SensitiveInt, or SensitiveFloat.")

    @overload
    def __rsub__(self, other: integer) -> SensitiveInt: ... # type: ignore[misc]
    @overload
    def __rsub__(self, other: floating) -> SensitiveFloat: ... # type: ignore[misc]

    def __rsub__(self, other: realnum) -> SensitiveInt | SensitiveFloat: # type: ignore[misc]
        if is_integer(other):
            return SensitiveInt(other - self._value, distance=self.distance, parents=[self])
        elif is_floating(other):
            return SensitiveFloat(other - self._value, distance=self.distance, parents=[self])
        else:
            raise ValueError("`other` must be a real number, SensitiveInt, or SensitiveFloat.")

    @overload
    def __mul__(self, other: integer) -> SensitiveInt: ...
    @overload
    def __mul__(self, other: floating) -> SensitiveFloat: ...

    def __mul__(self, other: realnum) -> SensitiveInt | SensitiveFloat:
        if is_integer(other):
            return SensitiveInt(self._value * other, distance=self.distance * _np.abs(other), parents=[self])
        elif is_floating(other):
            return SensitiveFloat(self._value * other, distance=self.distance * _np.abs(other), parents=[self])
        elif isinstance(other, (SensitiveInt, SensitiveFloat)):
            raise DPError("Sensitive values cannot be multiplied by each other.")
        else:
            raise ValueError("`other` must be a real number.")

    @overload
    def __rmul__(self, other: integer) -> SensitiveInt: ... # type: ignore[misc]
    @overload
    def __rmul__(self, other: floating) -> SensitiveFloat: ... # type: ignore[misc]

    def __rmul__(self, other: realnum) -> SensitiveInt | SensitiveFloat: # type: ignore[misc]
        if is_integer(other):
            return SensitiveInt(other * self._value, distance=self.distance * _np.abs(other), parents=[self])
        elif is_floating(other):
            return SensitiveFloat(other * self._value, distance=self.distance * _np.abs(other), parents=[self])
        else:
            raise ValueError("`other` must be a real number.")

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

    def __add__(self, other: realnum | SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        if is_realnum(other):
            return SensitiveFloat(self._value + other, distance=self.distance, parents=[self])
        elif isinstance(other, (SensitiveInt, SensitiveFloat)):
            return SensitiveFloat(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])
        else:
            raise ValueError("`other` must be a real number, SensitiveInt, or SensitiveFloat.")

    def __radd__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        if is_realnum(other):
            return SensitiveFloat(other + self._value, distance=self.distance, parents=[self])
        else:
            raise ValueError("`other` must be a real number, SensitiveInt, or SensitiveFloat.")

    def __sub__(self, other: realnum | SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        if is_realnum(other):
            return SensitiveFloat(self._value - other, distance=self.distance, parents=[self])
        elif isinstance(other, (SensitiveInt, SensitiveFloat)):
            return SensitiveFloat(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])
        else:
            raise ValueError("`other` must be a real number, SensitiveInt, or SensitiveFloat.")

    def __rsub__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        if is_realnum(other):
            return SensitiveFloat(other - self._value, distance=self.distance, parents=[self])
        else:
            raise ValueError("`other` must be a real number, SensitiveInt, or SensitiveFloat.")

    def __mul__(self, other: realnum) -> SensitiveFloat:
        if is_realnum(other):
            return SensitiveFloat(self._value * other, distance=self.distance * _np.abs(other), parents=[self])
        elif isinstance(other, (SensitiveInt, SensitiveFloat)):
            raise DPError("Sensitive values cannot be multiplied by each other.")
        else:
            raise ValueError("`other` must be a real number.")

    def __rmul__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        if is_realnum(other):
            return SensitiveFloat(other * self._value, distance=self.distance * _np.abs(other), parents=[self])
        else:
            raise ValueError("`other` must be a real number.")

@overload
def max2(a: SensitiveInt, b: SensitiveInt) -> SensitiveInt: ...
@overload
def max2(a: SensitiveInt, b: SensitiveFloat) -> SensitiveFloat: ...
@overload
def max2(a: SensitiveFloat, b: SensitiveInt) -> SensitiveFloat: ...
@overload
def max2(a: SensitiveFloat, b: SensitiveFloat) -> SensitiveFloat: ...

def max2(a: SensitiveInt | SensitiveFloat, b: SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat:
    if isinstance(a, SensitiveInt) and isinstance(b, SensitiveInt):
        return SensitiveInt(max(int(a._value), int(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])
    else:
        return SensitiveFloat(max(float(a._value), float(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

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

@overload
def min2(a: SensitiveInt, b: SensitiveInt) -> SensitiveInt: ...
@overload
def min2(a: SensitiveInt, b: SensitiveFloat) -> SensitiveFloat: ...
@overload
def min2(a: SensitiveFloat, b: SensitiveInt) -> SensitiveFloat: ...
@overload
def min2(a: SensitiveFloat, b: SensitiveFloat) -> SensitiveFloat: ...

def min2(a: SensitiveInt | SensitiveFloat, b: SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat:
    if isinstance(a, SensitiveInt) and isinstance(b, SensitiveInt):
        return SensitiveInt(min(int(a._value), int(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])
    else:
        return SensitiveFloat(min(float(a._value), float(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

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

def consumed_privacy_budget() -> dict[str, float]:
    return consumed_privacy_budget_all()
