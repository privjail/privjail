from __future__ import annotations
from typing import Any
from .util import realnum
import sympy as _sp # type: ignore[import-untyped]

class Distance:
    def __init__(self, expr: Any, constraints: Any = None):
        self.expr        = expr
        self.constraints = set(constraints) if constraints is not None else set()

    def __add__(self, other: realnum | Distance) -> Distance:
        if isinstance(other, Distance):
            return Distance(self.expr + other.expr, self.constraints.union(other.constraints))
        else:
            return Distance(self.expr + other, self.constraints)

    def __mul__(self, other: realnum | Distance) -> Distance:
        if isinstance(other, Distance):
            # TODO: should we allow distance * distance?
            return Distance(self.expr * other.expr, self.constraints.union(other.constraints))
        else:
            return Distance(self.expr * other, self.constraints)

    def max(self) -> realnum:
        return _sp.solvers.simplex.lpmax(self.expr, self.constraints)[0] # type: ignore[no-any-return]

    def is_zero(self) -> bool:
        return self.expr == 0 # type: ignore[no-any-return]

    def create_exclusive_distances(self, n_children: int) -> list[Distance]:
        # Create new child distance variables to express exclusiveness
        # d1 + d2 + ... + dn <= d_current
        dvars = [new_distance_var() for i in range(n_children)]
        constraints = {0 <= dvar for dvar in dvars} | \
                      {sum(dvars) <= self.expr} | \
                      self.constraints
        return [Distance(dvar, constraints) for dvar in dvars]

def new_distance_var() -> Any:
    return _sp.Dummy()
