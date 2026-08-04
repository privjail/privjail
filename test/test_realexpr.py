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

import importlib
import math
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
realexpr = importlib.import_module("privjail.realexpr")

def test_realexpr() -> None:
    d = realexpr.RealExpr(1)
    assert d.max() == 1

    d = d * 2
    assert d.max() == 2

    d = d + 1
    assert d.max() == 3

    x = realexpr.new_var()
    y = realexpr.new_var()
    z = realexpr.new_var()

    constraints = {realexpr.Constraint(frozenset({x, y, z}), 1)}

    dx = realexpr.RealExpr(x, constraints)
    dy = realexpr.RealExpr(y, constraints)
    dz = realexpr.RealExpr(z, constraints)

    assert dx.max() == 1

    d = dx + dy
    assert d.max() == 1

    d = d + dz
    assert d.max() == 1

    d = d * 2
    assert d.max() == 2

    d = d + 1
    assert d.max() == 3

    x_ = realexpr.new_var()
    y_ = realexpr.new_var()
    z_ = realexpr.new_var()

    constraints |= {realexpr.Constraint(frozenset({x_, y_, z_}), x)}

    dx_ = realexpr.RealExpr(x_, constraints)
    dy_ = realexpr.RealExpr(y_, constraints)
    dz_ = realexpr.RealExpr(z_, constraints)

    assert dx_.max() == 1

    d = dx_ + dy_ + dz_
    assert d.max() == 1

    d = d * 4
    assert d.max() == 4

    d = d + dx
    assert d.max() == 5

    d = d + dy + dz
    assert d.max() == 5

    d = d * 2
    assert d.max() == 10

    d = d + dx_
    assert d.max() == 11


def test_l2_constraint_is_evaluated_structurally() -> None:
    components = realexpr.RealExpr(2.0).create_l2_components(3)

    assert components[0].max() == pytest.approx(2.0)
    assert (components[0] + components[1] + components[2]).max() == (
        pytest.approx(2.0 * math.sqrt(3.0))
    )
    assert realexpr.joint_l2_max(components) == pytest.approx(2.0)
    assert realexpr.joint_l2_max(components[:2]) == pytest.approx(2.0)
    assert realexpr.joint_l2_max(
        [components[0], components[0]]
    ) == pytest.approx(2.0 * math.sqrt(2.0))


def test_max_does_not_mutate_the_expression_structure() -> None:
    expression = realexpr.RealExpr(2).create_exclusive_children(1)[0]
    original_expr = expression.expr
    original_constraints = set(expression.constraints)

    assert expression.max() == pytest.approx(2.0)
    assert expression.expr == original_expr
    assert expression.constraints == original_constraints


def test_joint_l2_max_combines_independent_constraints_in_quadrature() -> None:
    first = realexpr.RealExpr(1.0).create_l2_components(2)
    second = realexpr.RealExpr(2.0).create_l2_components(1)

    assert realexpr.joint_l2_max(
        [first[0], first[1], second[0]]
    ) == pytest.approx(math.sqrt(5.0))


def test_joint_l2_max_accounts_for_component_scaling() -> None:
    first, second = realexpr.RealExpr(2.0).create_l2_components(2)

    assert realexpr.joint_l2_max(
        [first * 2.0, second * 0.5]
    ) == pytest.approx(4.0)


def test_bind_l2_variables_preserves_sharing_without_reuse() -> None:
    first, second = realexpr.RealExpr(1.0).create_l2_components(2)
    mapping: dict[object, object] = {}

    bound_first = realexpr.bind_l2_variables(first, mapping)
    bound_second = realexpr.bind_l2_variables(second, mapping)

    assert first.l2_variables().isdisjoint(bound_first.l2_variables())
    assert bound_first.l2_variables() == bound_second.l2_variables()
    assert realexpr.joint_l2_max(
        [bound_first, bound_second]
    ) == pytest.approx(1.0)
