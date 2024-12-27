from __future__ import annotations
from typing import Literal

ChildrenType = Literal["inclusive", "exclusive"]

privacy_budget_count: int = 0

class ProvenanceEntity:
    parents                 : list[ProvenanceEntity]
    children_type           : ChildrenType
    root_name               : str
    consumed_privacy_budget : float
    depth                   : int

    def __init__(self, parents: list[ProvenanceEntity], children_type: ChildrenType, root_name: str | None = None):
        assert len(parents) > 0 or root_name is not None
        self.parents                 = parents
        self.children_type           = children_type
        self.root_name               = root_name if root_name is not None else parents[0].root_name
        self.consumed_privacy_budget = 0
        self.depth                   = max([p.depth for p in parents]) + 1 if len(parents) > 0 else 0

    def consume_privacy_budget(self, eps: float) -> None:
        assert eps >= 0
        self.consumed_privacy_budget += eps

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

def consume_privacy_budget(pes: list[ProvenanceEntity], eps: float) -> None:
    assert len(pes) > 0

    if are_exclusive_siblings(pes):
        for pe in pes:
            pe.consume_privacy_budget(eps)

        pe0 = pes[0].parents[0]
        new_cpd = max(pe0.consumed_privacy_budget, *[pe.consumed_privacy_budget for pe in pes])
        new_eps = new_cpd - pe0.consumed_privacy_budget

        if new_eps > 0:
            pe0.consume_privacy_budget(new_eps)
            consume_privacy_budget(pe0.parents, new_eps)

    elif len(pes) == 1 and len(pes[0].parents) == 0:
        assert get_provenance_root(pes[0].root_name) == pes[0]
        pes[0].consume_privacy_budget(eps)

    elif len(pes) == 1 and len(pes[0].parents) > 0:
        pes[0].consume_privacy_budget(eps)
        consume_privacy_budget(pes[0].parents, eps)

    else:
        # skip intermediate entities until all paths converge into a single entity
        # (lowest "single" common ancestor (LSCA) in a dag)
        consume_privacy_budget([get_lsca(pes)], eps)

def consumed_privacy_budget(name: str) -> float:
    return get_provenance_root(name).consumed_privacy_budget

def consumed_privacy_budget_all() -> dict[str, float]:
    global provenance_roots
    return {name: pe.consumed_privacy_budget for name, pe in provenance_roots.items()}

def get_lsca(pes: list[ProvenanceEntity]) -> ProvenanceEntity:
    max_depth = max([p.depth for p in pes])
    pe_set = set(pes)
    for d in reversed(range(max_depth + 1)):
        if len(pe_set) == 1:
            return next(iter(pe_set))
        pe_set = {pp for p in pe_set for pp in p.parents if p.depth == d}
    raise RuntimeError

# should not be exposed
def clear_global_states() -> None:
    global provenance_roots
    provenance_roots = {}
