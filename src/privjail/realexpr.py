# Copyright 2025 TOYOTA MOTOR CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import Any, NamedTuple
from .util import realnum, is_realnum
import sympy as _sp # type: ignore[import-untyped]

Var = Any
Expr = Any

class Constraint(NamedTuple):
    # Constraint: d1 + d2 + ... + dn <= de
    lhs: frozenset[Var] # variables {d1, d2, ..., dn}
    rhs: Expr           # expression de

def free_dvars(constraint: Constraint) -> frozenset[Var]:
    return constraint.lhs | (constraint.rhs.free_symbols if not is_realnum(constraint.rhs) else set())

class RealExpr:
    def __init__(self, expr: Expr, constraints: set[Constraint] | None = None):
        self.expr        = expr
        self.constraints = constraints if constraints is not None else set()

    def __add__(self, other: realnum | RealExpr) -> RealExpr:
        if isinstance(other, RealExpr):
            return RealExpr(self.expr + other.expr, self.constraints | other.constraints)
        else:
            return RealExpr(self.expr + other, self.constraints)

    def __mul__(self, other: realnum) -> RealExpr:
        # TODO: disallow RealExpr * RealExpr
        return RealExpr(self.expr * other, self.constraints)

    def max(self) -> realnum:
        if is_realnum(self.expr):
            return self.expr

        self._cleanup()

        # aggregate a subexpression (d1 + d2 + ... + dn) to a single variable
        # if they do not appear in other constraints or expressions
        sp_constraints = []
        dvars = self.expr.free_symbols
        for c in self.constraints:
            unused_dvars = c.lhs - dvars
            for c2 in self.constraints - {c}:
                unused_dvars -= free_dvars(c2)

            sp_constraints.append(sum(c.lhs - unused_dvars) <= c.rhs)

        # Remove constraints like d1 <= d2
        sp_expr = self.expr
        while True:
            changed = False
            for c in sp_constraints:
                if isinstance(c.lhs, _sp.Symbol):
                    sp_expr = sp_expr.subs(c.lhs, c.rhs)
                    sp_constraints = [c2.subs(c.lhs, c.rhs) for c2 in sp_constraints if c2 != c]
                    changed = True
                    break
            if not changed:
                break

        sp_constraints += list({0 <= d for c in sp_constraints for d in c.free_symbols})

        # Solve by linear programming
        y = _sp.solvers.simplex.lpmax(sp_expr, sp_constraints)[0]
        assert y.is_number
        return int(y) if y.is_integer else float(y)

    def is_zero(self) -> bool:
        return self.expr == 0 # type: ignore[no-any-return]

    def create_exclusive_children(self, n_children: int) -> list[RealExpr]:
        # Create new child variables to express exclusiveness
        # d1 + d2 + ... + dn <= d_current
        dvars = [new_var() for i in range(n_children)]
        constraints = self.constraints | {Constraint(frozenset(dvars), self.expr)}
        return [RealExpr(dvar, constraints) for dvar in dvars]

    def _cleanup(self) -> None:
        # simplify the expression by substituting d1 + d2 + ... + dn in self.expr
        # with constraints d1 + d2 + ... + dn <= d to get self.expr = d
        prev_expr = None
        while prev_expr != self.expr:
            prev_expr = self.expr
            self.expr = self.expr.subs([(sum(c.lhs), c.rhs) for c in self.constraints])

        # remove unused constraints
        constraints = set()
        dvars = self.expr.free_symbols
        prev_dvars = None
        while prev_dvars != dvars:
            prev_dvars = dvars
            constraints = {c for c in self.constraints if not c.lhs.isdisjoint(dvars)}
            dvars = {d for c in constraints for d in free_dvars(c)}
        self.constraints = constraints

var_count = 0

def new_var() -> Var:
    global var_count
    var_count += 1
    return _sp.Symbol(f"d{var_count}")

def _max(a: RealExpr, b: RealExpr) -> RealExpr:
    expr = _sp.Max(a.expr, b.expr)
    if expr.has(_sp.Max):
        # sympy.solvers.solveset.NonlinearError happens at lpmax() if Max() is included in the expression,
        # so we remove Max() here. However, the below is a loose approximation for the max operator.
        # TODO: improve handling for Max()
        return RealExpr(a.expr + b.expr, a.constraints | b.constraints)
    else:
        return RealExpr(expr, a.constraints | b.constraints)
