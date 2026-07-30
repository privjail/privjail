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
from collections.abc import Sequence
from typing import Any, NamedTuple
import math

import numpy as _np

from .util import realnum, is_realnum

import sympy as _sp # type: ignore[import-untyped]

Var = Any
Expr = Any

class Constraint(NamedTuple):
    # Constraint: d1 + d2 + ... + dn <= de
    lhs: frozenset[Var] # variables {d1, d2, ..., dn}
    rhs: Expr           # expression de

class L2Constraint(NamedTuple):
    # Constraint: sqrt(d1**2 + d2**2 + ... + dn**2) <= de
    components: frozenset[Var]
    bound: Expr

DistanceConstraint = Constraint | L2Constraint

def _free_symbols(expr: Expr) -> frozenset[Var]:
    if is_realnum(expr):
        return frozenset()
    return frozenset(expr.free_symbols)

def free_dvars(constraint: DistanceConstraint) -> frozenset[Var]:
    if isinstance(constraint, L2Constraint):
        return constraint.components | _free_symbols(constraint.bound)
    return constraint.lhs | _free_symbols(constraint.rhs)

class RealExpr:
    INF: RealExpr

    def __init__(
        self,
        expr: Expr,
        constraints: set[DistanceConstraint] | None = None,
        inf: bool = False,
    ):
        self.expr        = expr
        self.constraints = constraints if constraints is not None else set()
        self.inf         = inf

    def __add__(self, other: realnum | RealExpr) -> RealExpr:
        if self.inf or _is_inf(other):
            return RealExpr.INF

        if isinstance(other, RealExpr):
            return RealExpr(self.expr + other.expr, self.constraints | other.constraints)
        else:
            return RealExpr(self.expr + other, self.constraints)

    def __mul__(self, other: realnum | RealExpr) -> RealExpr:
        if self.inf or _is_inf(other):
            return RealExpr.INF

        if isinstance(other, RealExpr):
            return RealExpr(self.expr * other.expr, self.constraints | other.constraints)
        else:
            return RealExpr(self.expr * other, self.constraints)

    def max(self) -> realnum:
        if self.inf:
            return math.inf

        if is_realnum(self.expr):
            return self.expr

        if any(isinstance(c, L2Constraint) for c in self.constraints):
            return _max_with_l2_constraints(self)

        cleaned = RealExpr(self.expr, set(self.constraints))
        cleaned._cleanup()

        # aggregate a subexpression (d1 + d2 + ... + dn) to a single variable
        # if they do not appear in other constraints or expressions
        sp_constraints = []
        dvars = cleaned.expr.free_symbols
        for c in cleaned.constraints:
            assert isinstance(c, Constraint)
            unused_dvars = c.lhs - dvars
            for c2 in cleaned.constraints - {c}:
                unused_dvars -= free_dvars(c2)

            sp_constraints.append(sum(c.lhs - unused_dvars) <= c.rhs)

        # Remove constraints like d1 <= d2
        sp_expr = cleaned.expr
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

        if sp_expr.is_number:
            return int(sp_expr) if sp_expr.is_integer else float(sp_expr)

        sp_constraints += list({0 <= d for c in sp_constraints for d in c.free_symbols})

        # Solve by linear programming
        y = _sp.solvers.simplex.lpmax(sp_expr, sp_constraints)[0]
        assert y.is_number
        return int(y) if y.is_integer else float(y)

    def is_zero(self) -> bool:
        return not self.inf and self.expr == 0

    def is_inf(self) -> bool:
        return self.inf

    def structurally_equal(self, other: RealExpr) -> bool:
        return (
            self.inf == other.inf
            and self.expr == other.expr
            and self.constraints == other.constraints
        )

    def create_exclusive_children(self, n_children: int) -> list[RealExpr]:
        if self.inf:
            raise ValueError

        # Create new child variables to express exclusiveness
        # d1 + d2 + ... + dn <= d_current
        dvars = [new_var() for i in range(n_children)]
        constraints = self.constraints | {Constraint(frozenset(dvars), self.expr)}
        return [RealExpr(dvar, constraints) for dvar in dvars]

    def create_l2_components(self, n_components: int) -> list[RealExpr]:
        if self.inf:
            raise ValueError
        if n_components < 0:
            raise ValueError("The number of L2 components must be non-negative.")

        dvars = [new_var() for _ in range(n_components)]
        constraints = self.constraints | {
            L2Constraint(frozenset(dvars), self.expr)
        }
        return [RealExpr(dvar, constraints) for dvar in dvars]

    def l2_variables(self) -> frozenset[Var]:
        return frozenset(
            variable
            for constraint in self.constraints
            if isinstance(constraint, L2Constraint)
            for variable in constraint.components
        )

    def semantic_key(self, l2_variable_slots: dict[Var, int]) -> tuple[Any, ...]:
        substitutions = {
            variable: _sp.Symbol(f"_l2_{slot}")
            for variable, slot in l2_variable_slots.items()
        }

        def expr_key(expr: Expr) -> str:
            if is_realnum(expr):
                return repr(expr)
            return str(_sp.srepr(expr.xreplace(substitutions)))

        constraint_keys: list[tuple[Any, ...]] = []
        for constraint in self.constraints:
            if isinstance(constraint, L2Constraint):
                constraint_keys.append(
                    (
                        "l2",
                        tuple(
                            sorted(
                                l2_variable_slots[variable]
                                for variable in constraint.components
                            )
                        ),
                        expr_key(constraint.bound),
                    )
                )
            else:
                constraint_keys.append(
                    (
                        "linear",
                        tuple(
                            sorted(
                                expr_key(variable)
                                for variable in constraint.lhs
                            )
                        ),
                        expr_key(constraint.rhs),
                    )
                )
        return (
            expr_key(self.expr),
            tuple(sorted(constraint_keys, key=repr)),
            self.inf,
        )

    def _cleanup(self) -> None:
        linear_constraints = {
            constraint
            for constraint in self.constraints
            if isinstance(constraint, Constraint)
        }

        # simplify the expression by substituting d1 + d2 + ... + dn in self.expr
        # with constraints d1 + d2 + ... + dn <= d to get self.expr = d
        prev_expr = None
        while prev_expr != self.expr:
            prev_expr = self.expr
            self.expr = self.expr.subs(
                [(sum(c.lhs), c.rhs) for c in linear_constraints]
            )

        # remove unused constraints
        constraints: set[DistanceConstraint] = set()
        dvars = self.expr.free_symbols
        prev_dvars = None
        while prev_dvars != dvars:
            prev_dvars = dvars
            constraints = {
                constraint
                for constraint in self.constraints
                if not (
                    (
                        constraint.components
                        if isinstance(constraint, L2Constraint)
                        else constraint.lhs
                    )
                ).isdisjoint(dvars)
            }
            dvars = {d for c in constraints for d in free_dvars(c)}
        self.constraints = constraints


def _linear_constraints(
    constraints: set[DistanceConstraint],
) -> set[Constraint]:
    return {
        constraint
        for constraint in constraints
        if isinstance(constraint, Constraint)
    }


def _absolute_linear_max(
    expr: Expr,
    constraints: set[DistanceConstraint],
) -> float:
    if is_realnum(expr):
        return abs(float(expr))
    linear = _linear_constraints(constraints)
    upper = float(RealExpr(expr, set(linear)).max())
    lower = float(RealExpr(-expr, set(linear)).max())
    return max(upper, lower)


def _l2_constraint_map(
    constraints: set[DistanceConstraint],
) -> dict[Var, L2Constraint] | None:
    result: dict[Var, L2Constraint] = {}
    for constraint in constraints:
        if not isinstance(constraint, L2Constraint):
            continue
        for variable in constraint.components:
            previous = result.setdefault(variable, constraint)
            if previous != constraint:
                return None
    return result


def _linear_l2_terms(
    expr: Expr,
    variables: Sequence[Var],
) -> tuple[Expr, dict[Var, Expr]] | None:
    if not variables:
        return expr, {}
    try:
        polynomial = _sp.Poly(_sp.expand(expr), *variables)
    except _sp.PolynomialError:
        return None
    if polynomial.total_degree() > 1:
        return None

    constant = polynomial.coeff_monomial(1)
    coefficients = {
        variable: polynomial.coeff_monomial(variable)
        for variable in variables
    }
    return constant, coefficients


def _constraint_bound_max(
    constraint: L2Constraint,
    constraints: set[DistanceConstraint],
) -> float:
    bound = float(
        RealExpr(
            constraint.bound,
            set(_linear_constraints(constraints)),
        ).max()
    )
    if not math.isfinite(bound) or bound < 0:
        raise ValueError(f"Invalid L2 constraint bound ({bound}).")
    return bound


def _max_with_l2_constraints(value: RealExpr) -> realnum:
    constraint_map = _l2_constraint_map(value.constraints)
    if constraint_map is None:
        return math.inf
    variables = sorted(constraint_map, key=_sp.default_sort_key)
    terms = _linear_l2_terms(value.expr, variables)
    if terms is None:
        substitutions = {
            variable: _constraint_bound_max(
                constraint,
                value.constraints,
            )
            for variable, constraint in constraint_map.items()
        }
        substituted = value.expr.subs(substitutions)
        return RealExpr(
            substituted,
            set(_linear_constraints(value.constraints)),
        ).max()

    constant, coefficients = terms
    maximum = _absolute_linear_max(constant, value.constraints)
    for constraint in {
        constraint_map[variable]
        for variable in variables
    }:
        coefficient_norm = math.sqrt(
            sum(
                _absolute_linear_max(
                    coefficients[variable],
                    value.constraints,
                )
                ** 2
                for variable in constraint.components
            )
        )
        maximum += (
            _constraint_bound_max(constraint, value.constraints)
            * coefficient_norm
        )
    return maximum


def joint_l2_max(values: Sequence[RealExpr]) -> realnum:
    if not values:
        raise ValueError("joint_l2_max requires at least one expression.")
    if any(value.inf for value in values):
        return math.inf

    constraints = set().union(*(value.constraints for value in values))
    constraint_map = _l2_constraint_map(constraints)
    if constraint_map is None:
        return math.inf
    variables = sorted(constraint_map, key=_sp.default_sort_key)

    grouped_rows: dict[L2Constraint, list[dict[Var, float]]] = {}
    independent_maxima: list[float] = []
    for value in values:
        terms = _linear_l2_terms(value.expr, variables)
        if terms is None:
            return math.sqrt(sum(float(item.max()) ** 2 for item in values))
        constant, coefficients = terms
        nonzero_variables = {
            variable
            for variable, coefficient in coefficients.items()
            if coefficient != 0
        }
        groups = {
            constraint_map[variable]
            for variable in nonzero_variables
        }
        constant_max = _absolute_linear_max(constant, value.constraints)
        if len(groups) > 1 or (groups and constant_max != 0):
            return math.sqrt(sum(float(item.max()) ** 2 for item in values))
        if not groups:
            independent_maxima.append(float(value.max()))
            continue

        constraint = next(iter(groups))
        grouped_rows.setdefault(constraint, []).append(
            {
                variable: _absolute_linear_max(
                    coefficients[variable],
                    value.constraints,
                )
                for variable in constraint.components
            }
        )

    squared_maximum = sum(maximum ** 2 for maximum in independent_maxima)
    for constraint, rows in grouped_rows.items():
        components = sorted(
            constraint.components,
            key=_sp.default_sort_key,
        )
        matrix = _np.asarray(
            [
                [row.get(component, 0.0) for component in components]
                for row in rows
            ],
            dtype=float,
        )
        operator_norm = float(_np.linalg.norm(matrix, ord=2))
        group_maximum = (
            _constraint_bound_max(constraint, constraints)
            * operator_norm
        )
        squared_maximum += group_maximum ** 2
    return math.sqrt(squared_maximum)


def bind_l2_variables(
    value: RealExpr,
    variable_mapping: dict[Var, Var] | None = None,
) -> RealExpr:
    """Bind cached L2 variables to one call, allocating missing variables."""
    if value.inf:
        return RealExpr.INF
    if not value.l2_variables():
        return value

    mapping = {} if variable_mapping is None else variable_mapping
    for variable in sorted(value.l2_variables(), key=_sp.default_sort_key):
        if variable not in mapping:
            mapping[variable] = new_var()

    def substitute(expr: Expr) -> Expr:
        if is_realnum(expr):
            return expr
        return expr.xreplace(mapping)

    constraints: set[DistanceConstraint] = set()
    for constraint in value.constraints:
        if isinstance(constraint, L2Constraint):
            constraints.add(
                L2Constraint(
                    frozenset(mapping[variable] for variable in constraint.components),
                    substitute(constraint.bound),
                )
            )
        else:
            constraints.add(
                Constraint(
                    frozenset(substitute(variable) for variable in constraint.lhs),
                    substitute(constraint.rhs),
                )
            )
    return RealExpr(substitute(value.expr), constraints)

RealExpr.INF = RealExpr(0, constraints=None, inf=True)

var_count = 0

def new_var() -> Var:
    global var_count
    var_count += 1
    return _sp.Symbol(f"d{var_count}")

def _is_inf(x: realnum | RealExpr) -> bool:
    return (
        (isinstance(x, RealExpr) and x.inf)
        or x == math.inf
    )

def _max(a: RealExpr, b: RealExpr) -> RealExpr:
    if a.is_inf() or b.is_inf():
        return RealExpr.INF

    expr = _sp.Max(a.expr, b.expr)
    if expr.has(_sp.Max):
        # sympy.solvers.solveset.NonlinearError happens at lpmax() if Max() is included in the expression,
        # so we remove Max() here. However, the below is a loose approximation for the max operator.
        # TODO: improve handling for Max()
        return RealExpr(a.expr + b.expr, a.constraints | b.constraints)
    else:
        return RealExpr(expr, a.constraints | b.constraints)

def _min(a: RealExpr, b: RealExpr) -> RealExpr:
    if a.is_inf():
        return b

    if b.is_inf():
        return a

    expr = _sp.Min(a.expr, b.expr)
    if expr.has(_sp.Min):
        return RealExpr(a.expr + b.expr, a.constraints | b.constraints)
    else:
        return RealExpr(expr, a.constraints | b.constraints)
