from __future__ import annotations
from typing import TypeVar, Generic, Any, overload, Iterable, cast, Sequence
from .util import DPError, integer, floating, realnum, is_integer, is_floating, is_realnum
from .provenance import ProvenanceEntity, new_provenance_root, new_provenance_node, get_privacy_budget_all, NewTagType, ChildrenType
from .distance import Distance, _max as dmax
import numpy as _np

T = TypeVar("T")

class Prisoner(Generic[T]):
    _value            : T
    distance          : Distance
    provenance_entity : ProvenanceEntity | None
    parents           : Sequence[Prisoner[Any]]

    def __init__(self,
                 value         : T,
                 distance      : Distance,
                 *,
                 parents       : Sequence[Prisoner[Any]] = [],
                 root_name     : str | None              = None,
                 tag_type      : NewTagType              = "none",
                 children_type : ChildrenType            = "inclusive",
                 ):
        self._value   = value
        self.distance = distance
        self.parents  = [parent for parent in parents if not parent.distance.is_zero()]

        if distance.is_zero():
            self.provenance_entity = None

        elif len(self.parents) == 0:
            if root_name is None:
                raise ValueError("Both parents and root_name are not specified.")

            self.provenance_entity = new_provenance_root(root_name)

        else:
            pe_parents = [parent.provenance_entity for parent in self.parents if parent.provenance_entity is not None]
            self.provenance_entity = new_provenance_node(pe_parents, tag_type, children_type)

    def __str__(self) -> str:
        return f"Prisoner({type(self._value)}, distance={self.distance.max()})"

    def __repr__(self) -> str:
        return f"Prisoner({type(self._value)}, distance={self.distance.max()})"

    def has_same_tag(self, other: Prisoner[Any]) -> bool:
        if self.provenance_entity is None and other.provenance_entity is None:
            return True
        elif self.provenance_entity is None or other.provenance_entity is None:
            return False
        else:
            return self.provenance_entity.has_same_tag(other.provenance_entity)

    def consume_privacy_budget(self, privacy_budget: float) -> None:
        assert self.provenance_entity is not None
        self.provenance_entity.accumulate_privacy_budget(privacy_budget)

    def root_name(self) -> str:
        assert self.provenance_entity is not None
        if self.provenance_entity.tag is not None:
            return self.provenance_entity.tag.name
        else:
            return self.parents[0].root_name()

class SensitiveInt(Prisoner[integer]):
    def __init__(self,
                 value         : integer,
                 distance      : Distance                = Distance(0),
                 *,
                 parents       : Sequence[Prisoner[Any]] = [],
                 root_name     : str | None              = None,
                 tag_type      : NewTagType              = "none",
                 children_type : ChildrenType            = "inclusive",
                 ):
        if not is_integer(value):
            raise ValueError("`value` must be int for SensitveInt.")
        super().__init__(value, distance, parents=parents, root_name=root_name, tag_type=tag_type, children_type=children_type)

    @overload
    def __add__(self, other: integer | SensitiveInt) -> SensitiveInt: ...
    @overload
    def __add__(self, other: floating | SensitiveFloat) -> SensitiveFloat: ...

    def __add__(self, other: realnum | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat:
        if is_integer(other):
            return SensitiveInt(self._value + other, distance=self.distance, parents=[self])
        elif is_floating(other):
            return SensitiveFloat(self._value + other, distance=self.distance, parents=[self])
        elif isinstance(other, SensitiveInt):
            return SensitiveInt(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])
        elif isinstance(other, SensitiveFloat):
            return SensitiveFloat(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])
        else:
            raise ValueError("`other` must be a real number, SensitiveInt, or SensitiveFloat.")

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

class SensitiveFloat(Prisoner[floating]):
    def __init__(self,
                 value         : floating,
                 distance      : Distance                = Distance(0),
                 *,
                 parents       : Sequence[Prisoner[Any]] = [],
                 root_name     : str | None              = None,
                 tag_type      : NewTagType              = "none",
                 children_type : ChildrenType            = "inclusive",
                 ):
        if not is_floating(value):
            raise ValueError("`value` must be float for SensitveFloat.")
        super().__init__(value, distance, parents=parents, root_name=root_name, tag_type=tag_type, children_type=children_type)

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

def current_privacy_budget() -> dict[str, float]:
    return get_privacy_budget_all()
