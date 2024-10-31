from __future__ import annotations
from typing import TypeVar, Generic, Any
from .provenance import ProvenanceEntity, new_provenance_root

T = TypeVar("T")

class Prisoner(Generic[T]):
    _value            : T
    sensitivity       : float
    provenance_entity : ProvenanceEntity

    def __init__(self,
                 value       : T,
                 sensitivity : float,
                 *,
                 parent      : Prisoner[Any] | None = None,
                 root_name   : str | None           = None,
                 ):
        self._value = value
        self.sensitivity = sensitivity

        if parent is None:
            if root_name is None:
                raise ValueError("Both parent and root_name is not specified")

            self.provenance_entity = new_provenance_root(root_name)
        else:
            self.provenance_entity = parent.provenance_entity.add_child("inclusive")

    def __str__(self) -> str:
        return f"Prisoner({type(self._value)}, sensitivity={self.sensitivity})"

    def __repr__(self) -> str:
        return f"Prisoner({type(self._value)}, sensitivity={self.sensitivity})"
