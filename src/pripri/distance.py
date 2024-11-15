from __future__ import annotations
from typing import Any
import sympy as _sp # type: ignore[import-untyped]

class Distance:
    def __init__(self, expr: Any, constraints: Any = None):
        self.expr        = expr
        self.constraints = set(constraints) if constraints is not None else set()

    def __add__(self, other: int | float | Distance) -> Distance:
        if isinstance(other, Distance):
            return Distance(self.expr + other.expr, self.constraints.union(other.constraints))
        else:
            return Distance(self.expr + other, self.constraints)

    def __mul__(self, other: int | float | Distance) -> Distance:
        if isinstance(other, Distance):
            # TODO: should we allow distance * distance?
            return Distance(self.expr * other.expr, self.constraints.union(other.constraints))
        else:
            return Distance(self.expr * other, self.constraints)

    def max(self) -> int | float:
        return _sp.solvers.simplex.lpmax(self.expr, self.constraints)[0] # type: ignore[no-any-return]

def new_distance_var() -> Any:
    return _sp.Dummy()
