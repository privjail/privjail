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
import sys
from pathlib import Path

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
