from __future__ import annotations
from typing import TypeVar, Generic, Any
from .provenance import ProvenanceEntity, new_provenance_root, get_privacy_budget_all

T = TypeVar("T")

class Prisoner(Generic[T]):
    _value            : T
    sensitivity       : float
    provenance_entity : ProvenanceEntity
    root_name         : str

    def __init__(self,
                 value       : T,
                 sensitivity : float,
                 *,
                 parent      : Prisoner[Any] | None = None,
                 root_name   : str | None           = None,
                 renew_tag   : bool                 = True,
                 ):
        self._value = value
        self.sensitivity = sensitivity

        if parent is None:
            if root_name is None:
                raise ValueError("Both parent and root_name is not specified")

            self.provenance_entity = new_provenance_root(root_name)
            self.root_name = root_name
        else:
            self.provenance_entity = parent.provenance_entity.add_child("inclusive")
            self.root_name = parent.root_name

        if renew_tag:
            self.provenance_entity.renew_tag()

    def __str__(self) -> str:
        return f"Prisoner({type(self._value)}, sensitivity={self.sensitivity})"

    def __repr__(self) -> str:
        return f"Prisoner({type(self._value)}, sensitivity={self.sensitivity})"

    def has_same_tag(self, other: Prisoner[Any]) -> bool:
        return self.provenance_entity.has_same_tag(other.provenance_entity)

    def consume_privacy_budget(self, privacy_budget: float) -> None:
        self.provenance_entity.accumulate_privacy_budget(privacy_budget)

def current_privacy_budget() -> dict[str, float]:
    return get_privacy_budget_all()
