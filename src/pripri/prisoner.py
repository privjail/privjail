from __future__ import annotations
from typing import TypeVar, Generic, Any
from .util import DPError, integer, floating, realnum, is_integer, is_floating, is_realnum
from .provenance import ProvenanceEntity, new_provenance_root, new_provenance_node, get_privacy_budget_all, NewTagType, ChildrenType
from .distance import Distance

T = TypeVar("T")

class Prisoner(Generic[T]):
    _value            : T
    distance          : Distance
    provenance_entity : ProvenanceEntity | None
    parents           : list[Prisoner[Any]]

    def __init__(self,
                 value         : T,
                 distance      : Distance,
                 *,
                 parents       : list[Prisoner[Any]] = [],
                 root_name     : str | None          = None,
                 tag_type      : NewTagType          = "none",
                 children_type : ChildrenType        = "inclusive",
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

class SensitiveRealNum(Prisoner[realnum]):
    def __init__(self,
                 value         : realnum,
                 distance      : Distance            = Distance(0),
                 *,
                 parents       : list[Prisoner[Any]] = [],
                 root_name     : str | None          = None,
                 tag_type      : NewTagType          = "none",
                 children_type : ChildrenType        = "inclusive",
                 ):
        if not is_realnum(value):
            raise ValueError("`value` must be a real number for SensitveRealNumber.")
        super().__init__(value, distance, parents=parents, root_name=root_name, tag_type=tag_type, children_type=children_type)

    def __add__(self, other: realnum | SensitiveRealNum) -> SensitiveRealNum:
        if is_realnum(other):
            return SensitiveRealNum(self._value + other, distance=self.distance, parents=[self])
        elif isinstance(other, SensitiveRealNum):
            return SensitiveRealNum(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])
        else:
            raise ValueError("`other` must be a real number or SensitiveRealNum.")

    def __radd__(self, other: realnum | SensitiveRealNum) -> SensitiveRealNum: # type: ignore[misc]
        if is_realnum(other):
            return SensitiveRealNum(self._value + other, distance=self.distance, parents=[self])
        elif isinstance(other, SensitiveRealNum):
            return SensitiveRealNum(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])
        else:
            raise ValueError("`other` must be a real number or SensitiveRealNum.")

    def __sub__(self, other: realnum | SensitiveRealNum) -> SensitiveRealNum:
        if is_realnum(other):
            return SensitiveRealNum(self._value - other, distance=self.distance, parents=[self])
        elif isinstance(other, SensitiveRealNum):
            return SensitiveRealNum(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])
        else:
            raise ValueError("`other` must be a real number or SensitiveRealNum.")

    def __rsub__(self, other: realnum | SensitiveRealNum) -> SensitiveRealNum: # type: ignore[misc]
        if is_realnum(other):
            return SensitiveRealNum(self._value - other, distance=self.distance, parents=[self])
        elif isinstance(other, SensitiveRealNum):
            return SensitiveRealNum(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])
        else:
            raise ValueError("`other` must be a real number or SensitiveRealNum.")

    def __mul__(self, other: realnum) -> SensitiveRealNum:
        if is_realnum(other):
            return SensitiveRealNum(self._value * other, distance=self.distance * other, parents=[self])
        elif isinstance(other, SensitiveRealNum):
            raise DPError("Sensitive values cannot be multiplied by each other.")
        else:
            raise ValueError("`other` must be a real number.")

    def __rmul__(self, other: realnum) -> SensitiveRealNum: # type: ignore[misc]
        if is_realnum(other):
            return SensitiveRealNum(self._value * other, distance=self.distance * other, parents=[self])
        elif isinstance(other, SensitiveRealNum):
            raise DPError("Sensitive values cannot be multiplied by each other.")
        else:
            raise ValueError("`other` must be a real number.")

class SensitiveInt(SensitiveRealNum):
    def __init__(self,
                 value         : integer,
                 distance      : Distance            = Distance(0),
                 *,
                 parents       : list[Prisoner[Any]] = [],
                 root_name     : str | None          = None,
                 tag_type      : NewTagType          = "none",
                 children_type : ChildrenType        = "inclusive",
                 ):
        if not is_integer(value):
            raise ValueError("`value` must be int for SensitveInt.")
        super().__init__(value, distance, parents=parents, root_name=root_name, tag_type=tag_type, children_type=children_type)

class SensitiveFloat(SensitiveRealNum):
    def __init__(self,
                 value         : floating,
                 distance      : Distance            = Distance(0),
                 *,
                 parents       : list[Prisoner[Any]] = [],
                 root_name     : str | None          = None,
                 tag_type      : NewTagType          = "none",
                 children_type : ChildrenType        = "inclusive",
                 ):
        if not is_floating(value):
            raise ValueError("`value` must be float for SensitveFloat.")
        super().__init__(value, distance, parents=parents, root_name=root_name, tag_type=tag_type, children_type=children_type)

def current_privacy_budget() -> dict[str, float]:
    return get_privacy_budget_all()
