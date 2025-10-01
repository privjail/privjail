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

from typing import Any
import pytest
import uuid
import math

from privjail.accountants import *

def test_pure_accountant() -> None:
    a0 = PureAccountant(budget_limit=1.0)
    a0.set_as_root(name=str(uuid.uuid4()))

    assert a0.budget_spent() == 0

    eps = 0.1

    a0.spend(eps)
    assert a0.budget_spent() == pytest.approx(eps)

    a0.spend(eps)
    assert a0.budget_spent() == pytest.approx(eps * 2)

    # parallel composition
    ap = PureParallelAccountant(parent=a0)
    ap1 = PureAccountant(parent=ap)
    ap2 = PureAccountant(parent=ap)

    ap1.spend(eps)
    ap2.spend(eps)

    assert ap1.budget_spent() == pytest.approx(eps)
    assert ap2.budget_spent() == pytest.approx(eps)
    assert a0.budget_spent() == pytest.approx(eps * 3)

    ap1.spend(eps)

    assert ap1.budget_spent() == pytest.approx(eps * 2)
    assert ap2.budget_spent() == pytest.approx(eps)
    assert a0.budget_spent() == pytest.approx(eps * 4)

    with pytest.raises(BudgetExceededError):
        ap1.spend(1.0)

    # exceeding budget should not be counted
    assert ap1.budget_spent() == pytest.approx(eps * 2)
    assert ap2.budget_spent() == pytest.approx(eps)
    assert a0.budget_spent() == pytest.approx(eps * 4)

    with pytest.raises(BudgetExceededError):
        a0.spend(eps * 7)

    assert a0.budget_spent() == pytest.approx(eps * 4)

    with pytest.raises(Exception):
        a0.spend(-1.0)

def test_approx_accountant() -> None:
    a0 = ApproxAccountant(budget_limit=(1.0, 1e-6))
    a0.set_as_root(name=str(uuid.uuid4()))

    assert a0.budget_spent() == (0, 0)

    eps = 0.1
    delta = 1e-7

    a0.spend((eps, 0))
    assert a0.budget_spent() == pytest.approx((eps, 0))

    a0.spend((eps, delta))
    assert a0.budget_spent() == pytest.approx((eps * 2, delta))

    # parallel composition
    ap = ApproxParallelAccountant(parent=a0)
    ap1 = ApproxAccountant(parent=ap)
    ap2 = ApproxAccountant(parent=ap)

    ap1.spend((eps, delta))
    ap2.spend((eps, delta))

    assert ap1.budget_spent() == pytest.approx((eps, delta))
    assert ap2.budget_spent() == pytest.approx((eps, delta))
    assert a0.budget_spent() == pytest.approx((eps * 3, delta * 2))

    ap1.spend((eps, 0))
    ap2.spend((0, delta))

    assert ap1.budget_spent() == pytest.approx((eps * 2, delta))
    assert ap2.budget_spent() == pytest.approx((eps, delta * 2))
    assert a0.budget_spent() == pytest.approx((eps * 4, delta * 3))

    with pytest.raises(BudgetExceededError):
        ap1.spend((1.0, 0))

    with pytest.raises(BudgetExceededError):
        ap1.spend((0, delta * 10))

    # exceeding budget should not be counted
    assert ap1.budget_spent() == pytest.approx((eps * 2, delta))
    assert ap2.budget_spent() == pytest.approx((eps, delta * 2))
    assert a0.budget_spent() == pytest.approx((eps * 4, delta * 3))

    with pytest.raises(Exception):
        a0.spend((-1.0, -delta))

    with pytest.raises(Exception):
        a0.spend((-1.0, 0))

    with pytest.raises(Exception):
        a0.spend((0, -delta))

    # pure -> approx
    a_pure = PureAccountant(parent=a0)

    a_pure.spend(eps)

    assert a_pure.budget_spent() == pytest.approx(eps)
    assert a0.budget_spent() == pytest.approx((eps * 5, delta * 3))

    a_pure.spend(eps)

    assert a_pure.budget_spent() == pytest.approx(eps * 2)
    assert a0.budget_spent() == pytest.approx((eps * 6, delta * 3))

def test_zCDP_accountant() -> None:
    delta = 1e-6

    a0 = ApproxAccountant(budget_limit=(1.0, delta))
    a0.set_as_root(name=str(uuid.uuid4()))

    assert a0.budget_spent() == (0, 0)

    # should raise error without delta
    with pytest.raises(Exception):
        zCDPAccountant(parent=a0)

    a_zcdp = zCDPAccountant(parent=a0, delta=delta)

    assert a_zcdp.budget_spent() == 0

    rho = 0.001

    def eps_expected(rho: float, delta: float) -> float:
        return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))

    a_zcdp.spend(rho)

    assert a_zcdp.budget_spent() == pytest.approx(rho)
    assert a0.budget_spent() == pytest.approx((eps_expected(rho, delta), delta))

    a_zcdp.spend(rho)

    assert a_zcdp.budget_spent() == pytest.approx(rho * 2)
    assert a0.budget_spent() == pytest.approx((eps_expected(rho * 2, delta), delta))

    a_zcdp_p = zCDPParallelAccountant(parent=a_zcdp)
    a_zcdp_p1 = zCDPAccountant(parent=a_zcdp_p)
    a_zcdp_p2 = zCDPAccountant(parent=a_zcdp_p)

    a_zcdp_p1.spend(rho)
    a_zcdp_p2.spend(rho)

    assert a_zcdp_p1.budget_spent() == pytest.approx(rho)
    assert a_zcdp_p2.budget_spent() == pytest.approx(rho)
    assert a_zcdp.budget_spent() == pytest.approx(rho * 3)
    assert a0.budget_spent() == pytest.approx((eps_expected(rho * 3, delta), delta))

    a_zcdp_p1.spend(rho)

    assert a_zcdp_p1.budget_spent() == pytest.approx(rho * 2)
    assert a_zcdp_p2.budget_spent() == pytest.approx(rho)
    assert a_zcdp.budget_spent() == pytest.approx(rho * 4)
    assert a0.budget_spent() == pytest.approx((eps_expected(rho * 4, delta), delta))

    with pytest.raises(BudgetExceededError):
        a_zcdp_p1.spend(0.1)

    # exceeding budget should not be counted
    assert a_zcdp_p1.budget_spent() == pytest.approx(rho * 2)
    assert a_zcdp_p2.budget_spent() == pytest.approx(rho)
    assert a_zcdp.budget_spent() == pytest.approx(rho * 4)
    assert a0.budget_spent() == pytest.approx((eps_expected(rho * 4, delta), delta))

    with pytest.raises(Exception):
        a_zcdp.spend(-1.0)
