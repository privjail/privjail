from __future__ import annotations
from typing import NamedTuple, Literal

class ProvenanceTag(NamedTuple):
    name : str
    cnt  : int

NewTagType   = Literal["inherit", "renew", "none"]
ChildrenType = Literal["inclusive", "exclusive"]

privacy_budget_count: int = 0

class ProvenanceEntity:
    parents         : list[ProvenanceEntity]
    tag             : ProvenanceTag | None
    children_type   : ChildrenType
    privacy_budgets : dict[int, float]

    def __init__(self, parents: list[ProvenanceEntity], tag: ProvenanceTag | None, children_type: ChildrenType):
        self.parents         = parents
        self.tag             = tag
        self.children_type   = children_type
        self.privacy_budgets = {}

    def total_privacy_budget(self) -> float:
        return sum(self.privacy_budgets.values())

    def update_privacy_budget(self, delta: float, child_budget: float, budget_id: int) -> None:
        if budget_id in self.privacy_budgets and self.privacy_budgets[budget_id] >= delta:
            return

        if self.children_type == "inclusive":
            self.privacy_budgets[budget_id] = delta

            for parent in self.parents:
                parent.update_privacy_budget(delta, self.total_privacy_budget(), budget_id)

        elif self.children_type == "exclusive":
            prev_total_budget = self.total_privacy_budget()
            new_total_budget = max(prev_total_budget, child_budget)

            self.privacy_budgets[budget_id] = new_total_budget - prev_total_budget

            for parent in self.parents:
                parent.update_privacy_budget(new_total_budget - prev_total_budget, new_total_budget, budget_id)

        else:
            raise RuntimeError

    def accumulate_privacy_budget(self, privacy_budget: float) -> None:
        global privacy_budget_count

        assert self.children_type == "inclusive"

        budget_id = privacy_budget_count
        privacy_budget_count += 1

        self.update_privacy_budget(privacy_budget, 0, budget_id)

    def has_same_tag(self, other: ProvenanceEntity) -> bool:
        return self.tag == other.tag

provenance_roots      : dict[str, ProvenanceEntity] = {}
provenance_tag_counts : dict[str, int]              = {}

def new_provenance_root(name: str) -> ProvenanceEntity:
    global provenance_roots
    global provenance_tag_counts

    if name in provenance_roots:
        raise ValueError(f"Name '{name}' already exists")

    initial_tag_count = 0
    provenance_tag_counts[name] = initial_tag_count

    pe = ProvenanceEntity([], ProvenanceTag(name, initial_tag_count), "inclusive")
    provenance_roots[name] = pe

    return pe

def new_provenance_node(parents: list[ProvenanceEntity], tag_type: NewTagType, children_type: ChildrenType) -> ProvenanceEntity:
    assert len(parents) > 0
    assert all([pe.tag == parents[0].tag for pe in parents])

    if tag_type == "inherit":
        tag = parents[0].tag
    elif tag_type == "renew":
        tag = renew_tag(parents[0].tag)
    elif tag_type == "none":
        tag = None
    else:
        raise RuntimeError

    return ProvenanceEntity(parents, tag, children_type)

def renew_tag(tag: ProvenanceTag | None) -> ProvenanceTag:
    global provenance_tag_counts

    if tag is None:
        raise RuntimeError

    name, count = tag

    assert name in provenance_tag_counts
    provenance_tag_counts[name] += 1

    return ProvenanceTag(name, provenance_tag_counts[name])

def get_privacy_budget(name: str) -> float:
    global provenance_roots

    if name not in provenance_roots:
        raise ValueError(f"Name '{name}' does not exist")

    return provenance_roots[name].total_privacy_budget()

def get_privacy_budget_all() -> dict[str, float]:
    global provenance_roots

    return {name: pe.total_privacy_budget() for name, pe in provenance_roots.items()}

# should not be exposed
def clear_global_states() -> None:
    global provenance_roots
    global provenance_tag_counts

    provenance_roots = {}
    provenance_tag_counts = {}
