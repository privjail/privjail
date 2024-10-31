from __future__ import annotations
from typing import NamedTuple, Literal

class ProvenanceTag(NamedTuple):
    name : str
    cnt  : int

provenance_tag_counts: dict[str, int] = {}

ChildrenType = Literal["inclusive", "exclusive"]

class ProvenanceEntity:
    parent               : ProvenanceEntity | None
    tag                  : ProvenanceTag
    children_type        : ChildrenType
    privacy_budget_local : float
    privacy_budget       : float
    children             : list[ProvenanceEntity]

    def __init__(self, parent: ProvenanceEntity | None, tag: ProvenanceTag, children_type: ChildrenType):
        self.parent               = parent
        self.tag                  = tag
        self.children_type        = children_type
        self.privacy_budget_local = 0
        self.privacy_budget       = 0
        self.children             = []

    def update_privacy_budget(self) -> None:
        if self.children_type == "inclusive":
            self.privacy_budget = self.privacy_budget_local + sum([c.privacy_budget for c in self.children])

            if self.parent is not None:
                self.parent.update_privacy_budget()

        elif self.children_type == "exclusive":
            self.privacy_budget = self.privacy_budget_local + max([0] + [c.privacy_budget for c in self.children])

            if self.parent is not None:
                self.parent.update_privacy_budget()

        else:
            raise RuntimeError

    def accumulate_privacy_budget(self, privacy_budget: float) -> None:
        assert self.children_type == "inclusive"

        self.privacy_budget_local += privacy_budget

        self.update_privacy_budget()

    def add_child(self, children_type: ChildrenType) -> ProvenanceEntity:
        pe = ProvenanceEntity(self, self.tag, children_type)
        self.children.append(pe)
        return pe

    def renew_tag(self) -> None:
        global provenance_tag_counts

        name, count = self.tag

        assert name in provenance_tag_counts
        provenance_tag_counts[name] += 1

        self.tag = ProvenanceTag(name, provenance_tag_counts[name])

    def has_same_tag(self, other: ProvenanceEntity) -> bool:
        return self.tag == other.tag

provenance_roots: dict[str, ProvenanceEntity] = {}

def new_provenance_root(name: str) -> ProvenanceEntity:
    global provenance_roots
    global provenance_tag_counts

    if name in provenance_roots:
        raise ValueError(f"Name '{name}' already exists")

    initial_tag_count = 0
    provenance_tag_counts[name] = initial_tag_count

    pe = ProvenanceEntity(None, ProvenanceTag(name, initial_tag_count), "inclusive")
    provenance_roots[name] = pe

    return pe

def get_privacy_budget(name: str) -> float:
    global provenance_roots

    if name not in provenance_roots:
        raise ValueError(f"Name '{name}' does not exist")

    return provenance_roots[name].privacy_budget

def get_privacy_budget_all() -> dict[str, float]:
    global provenance_roots

    return {name: pe.privacy_budget for name, pe in provenance_roots.items()}

# should not be exposed
def clear_global_states() -> None:
    global provenance_roots
    global provenance_tag_counts

    provenance_roots = {}
    provenance_tag_counts = {}
