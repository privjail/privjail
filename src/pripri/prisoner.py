from __future__ import annotations
from typing import TypeVar, Generic, Any
from .provenance import ProvenanceEntity, new_provenance_root, new_provenance_node, get_privacy_budget_all, NewTagType, ChildrenType
from .distance import Distance

T = TypeVar("T")

class Prisoner(Generic[T]):
    _value            : T
    distance          : Distance
    provenance_entity : ProvenanceEntity
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
        self.parents  = parents

        if len(parents) == 0:
            if root_name is None:
                raise ValueError("Both parents and root_name are not specified.")

            self.provenance_entity = new_provenance_root(root_name)
        else:
            pe_parents = [parent.provenance_entity for parent in parents]
            self.provenance_entity = new_provenance_node(pe_parents, tag_type, children_type)

    def __str__(self) -> str:
        return f"Prisoner({type(self._value)}, distance={self.distance.max()})"

    def __repr__(self) -> str:
        return f"Prisoner({type(self._value)}, distance={self.distance.max()})"

    def has_same_tag(self, other: Prisoner[Any]) -> bool:
        return self.provenance_entity.has_same_tag(other.provenance_entity)

    def consume_privacy_budget(self, privacy_budget: float) -> None:
        self.provenance_entity.accumulate_privacy_budget(privacy_budget)

    def root_name(self) -> str:
        if self.provenance_entity.tag is not None:
            return self.provenance_entity.tag.name
        else:
            return self.parents[0].root_name()

def current_privacy_budget() -> dict[str, float]:
    return get_privacy_budget_all()
