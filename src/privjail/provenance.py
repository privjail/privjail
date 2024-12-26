from __future__ import annotations
from typing import Literal

ChildrenType = Literal["inclusive", "exclusive"]

privacy_budget_count: int = 0

class ProvenanceEntity:
    parents                 : list[ProvenanceEntity]
    children_type           : ChildrenType
    root_name               : str
    consumed_privacy_budget : float

    def __init__(self, parents: list[ProvenanceEntity], children_type: ChildrenType, root_name: str | None = None):
        assert len(parents) > 0 or root_name is not None
        self.parents                 = parents
        self.children_type           = children_type
        self.root_name               = root_name if root_name is not None else parents[0].root_name
        self.consumed_privacy_budget = 0

    def consume_privacy_budget(self, epsilon: float) -> None:
        assert epsilon >= 0
        self.consumed_privacy_budget += epsilon

provenance_roots : dict[str, ProvenanceEntity] = {}

def new_provenance_root(name: str) -> ProvenanceEntity:
    global provenance_roots

    if name in provenance_roots:
        raise ValueError(f"Name '{name}' already exists")

    pe = ProvenanceEntity([], "inclusive", root_name=name)
    provenance_roots[name] = pe

    return pe

def new_provenance_node(parents: list[ProvenanceEntity], children_type: ChildrenType) -> ProvenanceEntity:
    assert len(parents) > 0
    return ProvenanceEntity(parents, children_type)

def get_provenance_root(name: str) -> ProvenanceEntity:
    global provenance_roots

    if name not in provenance_roots:
        raise ValueError(f"Name '{name}' does not exist")

    return provenance_roots[name]

def are_exclusive_siblings(pes: list[ProvenanceEntity]) -> bool:
    assert len(pes) > 0
    return all([len(pe.parents) == 1 and \
                pes[0].parents[0] == pe.parents[0] and \
                pe.parents[0].children_type == "exclusive" for pe in pes])

def consume_privacy_budget(pes: list[ProvenanceEntity], epsilon: float) -> None:
    assert len(pes) > 0

    if are_exclusive_siblings(pes):
        for pe in pes:
            pe.consume_privacy_budget(epsilon)

        pe0 = pes[0].parents[0]
        new_cpd = max(pe0.consumed_privacy_budget, *[pe.consumed_privacy_budget for pe in pes])
        new_epsilon = new_cpd - pe0.consumed_privacy_budget

        if new_epsilon > 0:
            pe0.consume_privacy_budget(new_epsilon)
            consume_privacy_budget(pe0.parents, new_epsilon)

    elif len(pes) == 1 and len(pes[0].parents) == 0:
        assert get_provenance_root(pes[0].root_name) == pes[0]
        pes[0].consume_privacy_budget(epsilon)

    elif len(pes) == 1 and len(pes[0].parents) > 0:
        pes[0].consume_privacy_budget(epsilon)
        consume_privacy_budget(pes[0].parents, epsilon)

    else:
        # skip all intermediate entities to reach the root entry
        # TODO: find a least common ancestor and write tests for it
        assert all([pes[0].root_name == pe.root_name for pe in pes])
        get_provenance_root(pes[0].root_name).consume_privacy_budget(epsilon)

def consumed_privacy_budget(name: str) -> float:
    return get_provenance_root(name).consumed_privacy_budget

def consumed_privacy_budget_all() -> dict[str, float]:
    global provenance_roots
    return {name: pe.consumed_privacy_budget for name, pe in provenance_roots.items()}

# should not be exposed
def clear_global_states() -> None:
    global provenance_roots
    provenance_roots = {}
