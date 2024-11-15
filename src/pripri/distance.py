from typing import Any
import sympy as _sp # type: ignore[import-untyped]

class Distance:
    def __init__(self, expr: Any, constraints: Any = None):
        self.expr        = expr
        self.constraints = constraints if constraints is not None else []

    def max(self) -> int | float:
        return _sp.solvers.simplex.lpmax(self.expr, self.constraints)[0] # type: ignore[no-any-return]

def new_distance_var() -> Any:
    return _sp.Dummy()
