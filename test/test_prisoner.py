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

import pytest
import uuid
import privjail as pj
from privjail.accountants import ApproxAccountant

def new_sensitive_int(value: int) -> pj.SensitiveInt:
    # TODO: how to handle multimethod function types?
    accountant = pj.PureAccountant()
    accountant.set_as_root(name=str(uuid.uuid4()))
    return pj.SensitiveInt(value, distance=pj.RealExpr(1), accountant=accountant) + 0 # type: ignore[no-any-return]

def test_sensitive_real_number() -> None:
    x = new_sensitive_int(12)
    assert isinstance(x, pj.SensitiveInt)

    y = x * (1 / 12)
    assert isinstance(y, pj.SensitiveFloat)

    x = x + 1
    assert x._value == 13
    assert x.distance.max() == 1

    x = 1 + x
    assert x._value == 14
    assert x.distance.max() == 1

    x += 1
    assert x._value == 15
    assert x.distance.max() == 1

    x *= 3
    assert x._value == 45
    assert x.distance.max() == 3

    x = x * 2 + 1
    assert x._value == 91
    assert x.distance.max() == 6

    x -= 90
    assert x._value == 1
    assert x.distance.max() == 6

    with pytest.raises(TypeError):
        x * x

    z = x + y
    assert z._value == pytest.approx(2.0)
    assert z.distance.max() == pytest.approx(6.0 + 1 / 12)

    z = x - 2 * y
    assert z._value == pytest.approx(-1.0)
    assert z.distance.max() == pytest.approx(6.0 + 1 / 6)

    with pytest.raises(TypeError):
        x * y

def test_min_max() -> None:
    x = new_sensitive_int(11)
    y = x - 1
    z = x + 2

    assert pj.max(x, y)._value == 11
    assert pj.max(x, y).distance.max() == 1

    assert pj.min(x, y)._value == 10
    assert pj.min(x, y).distance.max() == 1

    assert pj.max(x, y, z)._value == 13
    assert pj.max(x, y, z).distance.max() == 1

    assert pj.min(x, y, z)._value == 10
    assert pj.min(x, y, z).distance.max() == 1

    z *= 2

    assert pj.max(x, y, z)._value == 26
    assert pj.max(x, y, z).distance.max() == 2

    assert pj.min(x, y, z)._value == 10
    assert pj.min(x, y, z).distance.max() == 2

    assert pj.max([x, y, z])._value == 26
    assert pj.max([x, y, z]).distance.max() == 2

    assert pj.min([x, y, z])._value == 10
    assert pj.min([x, y, z]).distance.max() == 2

    assert isinstance(pj.max(x * 3.0, y, z), pj.SensitiveFloat)
    assert pj.max(x * 3.0, y, z)._value == pytest.approx(33.0)
    assert pj.max(x * 3.0, y, z).distance.max() == pytest.approx(3.0)

    assert isinstance(pj.min(x * 3.0, y, z), pj.SensitiveFloat)
    assert pj.min(x * 3.0, y, z)._value == pytest.approx(10.0)
    assert pj.min(x * 3.0, y, z).distance.max() == pytest.approx(3.0)

    assert isinstance(pj.max([x, y, z - 25.0]), pj.SensitiveFloat)
    assert pj.max([x, y, z - 25.0])._value == pytest.approx(11.0)
    assert pj.max([x, y, z - 25.0]).distance.max() == pytest.approx(2.0)

    assert isinstance(pj.min([x, y, z - 25.0]), pj.SensitiveFloat)
    assert pj.min([x, y, z - 25.0])._value == pytest.approx(1.0)
    assert pj.min([x, y, z - 25.0]).distance.max() == pytest.approx(2.0)

    with pytest.raises(TypeError): pj.max()
    with pytest.raises(TypeError): pj.min()

    with pytest.raises(TypeError): pj.max(x)
    with pytest.raises(TypeError): pj.min(x)

    with pytest.raises(ValueError): pj.max([])
    with pytest.raises(ValueError): pj.min([])

def test_rdp_parent_budget_limit() -> None:
    # RDP->Approx conversion (sensitivity=1.0, scale=1.0, delta=1e-6):
    # ε_rdp(α) = α/2, ε_approx = min_α(ε_rdp(α) + log(1/δ)/(α-1))
    # After 1st call: min(1+13.82, 2+4.61, 4+1.97) ≈ 5.97
    # After 2nd call: min(2+13.82, 4+4.61, 8+1.97) ≈ 8.61
    # With parent_budget_limit=(8.0, 1e-6), 1st call OK, 2nd call raises error
    accountant = ApproxAccountant(budget_limit=(100.0, 1e-4))
    accountant.set_as_root(name=str(uuid.uuid4()))
    x = pj.SensitiveFloat(1.0, distance=pj.RealExpr(1), accountant=accountant)

    with pj.RDP(x, alpha=[2, 4, 8], parent_budget_limit=(8.0, 1e-6)) as x_rdp:
        assert isinstance(x_rdp, pj.SensitiveFloat)
        assert x_rdp.accountant.parent is not None
        assert x_rdp.accountant.parent._budget_limit == (8.0, 1e-6)

        pj.gaussian_mechanism(x_rdp, scale=1.0)
        eps_spent = x_rdp.accountant.parent.budget_spent()[0]
        assert 5.9 < eps_spent < 6.0

        with pytest.raises(pj.BudgetExceededError):
            pj.gaussian_mechanism(x_rdp, scale=1.0)
