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
import math

from privjail.accountants import *
from privjail.util import DPError

def test_pure_accountant() -> None:
    a0 = PureDPAccountant(budget_limit=1.0)
    a0.set_as_root(name=str(uuid.uuid4()))

    assert a0.budget_spent == 0

    eps = 0.1

    a0.spend(eps)
    assert a0.budget_spent == pytest.approx(eps)

    a0.spend(eps)
    assert a0.budget_spent == pytest.approx(eps * 2)

    # parallel composition
    ap = PureDPParallelAccountant(parent=a0)
    ap1 = PureDPAccountant(parent=ap)
    ap2 = PureDPAccountant(parent=ap)

    ap1.spend(eps)
    ap2.spend(eps)

    assert ap1.budget_spent == pytest.approx(eps)
    assert ap2.budget_spent == pytest.approx(eps)
    assert a0.budget_spent == pytest.approx(eps * 3)

    ap1.spend(eps)

    assert ap1.budget_spent == pytest.approx(eps * 2)
    assert ap2.budget_spent == pytest.approx(eps)
    assert a0.budget_spent == pytest.approx(eps * 4)

    with pytest.raises(BudgetExceededError):
        ap1.spend(1.0)

    # exceeding budget should not be counted
    assert ap1.budget_spent == pytest.approx(eps * 2)
    assert ap2.budget_spent == pytest.approx(eps)
    assert a0.budget_spent == pytest.approx(eps * 4)

    with pytest.raises(BudgetExceededError):
        a0.spend(eps * 7)

    assert a0.budget_spent == pytest.approx(eps * 4)

    with pytest.raises(Exception):
        a0.spend(-1.0)

def test_approx_accountant() -> None:
    a0 = ApproxDPAccountant(budget_limit=(1.0, 1e-6))
    a0.set_as_root(name=str(uuid.uuid4()))

    assert a0.budget_spent == (0, 0)

    eps = 0.1
    delta = 1e-7

    a0.spend((eps, 0))
    assert a0.budget_spent == pytest.approx((eps, 0))

    a0.spend((eps, delta))
    assert a0.budget_spent == pytest.approx((eps * 2, delta))

    # parallel composition
    ap = ApproxDPParallelAccountant(parent=a0)
    ap1 = ApproxDPAccountant(parent=ap)
    ap2 = ApproxDPAccountant(parent=ap)

    ap1.spend((eps, delta))
    ap2.spend((eps, delta))

    assert ap1.budget_spent == pytest.approx((eps, delta))
    assert ap2.budget_spent == pytest.approx((eps, delta))
    assert a0.budget_spent == pytest.approx((eps * 3, delta * 2))

    ap1.spend((eps, 0))
    ap2.spend((0, delta))

    assert ap1.budget_spent == pytest.approx((eps * 2, delta))
    assert ap2.budget_spent == pytest.approx((eps, delta * 2))
    assert a0.budget_spent == pytest.approx((eps * 4, delta * 3))

    with pytest.raises(BudgetExceededError):
        ap1.spend((1.0, 0))

    with pytest.raises(BudgetExceededError):
        ap1.spend((0, delta * 10))

    # exceeding budget should not be counted
    assert ap1.budget_spent == pytest.approx((eps * 2, delta))
    assert ap2.budget_spent == pytest.approx((eps, delta * 2))
    assert a0.budget_spent == pytest.approx((eps * 4, delta * 3))

    with pytest.raises(Exception):
        a0.spend((-1.0, -delta))

    with pytest.raises(Exception):
        a0.spend((-1.0, 0))

    with pytest.raises(Exception):
        a0.spend((0, -delta))

    # pure -> approx
    a_pure = PureDPAccountant(parent=a0)

    a_pure.spend(eps)

    assert a_pure.budget_spent == pytest.approx(eps)
    assert a0.budget_spent == pytest.approx((eps * 5, delta * 3))

    a_pure.spend(eps)

    assert a_pure.budget_spent == pytest.approx(eps * 2)
    assert a0.budget_spent == pytest.approx((eps * 6, delta * 3))

def test_zCDP_accountant() -> None:
    delta = 1e-6

    a0 = ApproxDPAccountant(budget_limit=(1.0, delta))
    a0.set_as_root(name=str(uuid.uuid4()))

    assert a0.budget_spent == (0, 0)

    # should raise error without delta
    with pytest.raises(Exception):
        zCDPAccountant(parent=a0)

    a_zcdp = zCDPAccountant(parent=a0, delta=delta)

    assert a_zcdp.budget_spent == 0

    rho = 0.001

    def eps_expected(rho: float, delta: float) -> float:
        return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))

    a_zcdp.spend(rho)

    assert a_zcdp.budget_spent == pytest.approx(rho)
    assert a0.budget_spent == pytest.approx((eps_expected(rho, delta), delta))

    a_zcdp.spend(rho)

    assert a_zcdp.budget_spent == pytest.approx(rho * 2)
    assert a0.budget_spent == pytest.approx((eps_expected(rho * 2, delta), delta))

    a_zcdp_p = zCDPParallelAccountant(parent=a_zcdp)
    a_zcdp_p1 = zCDPAccountant(parent=a_zcdp_p)
    a_zcdp_p2 = zCDPAccountant(parent=a_zcdp_p)

    a_zcdp_p1.spend(rho)
    a_zcdp_p2.spend(rho)

    assert a_zcdp_p1.budget_spent == pytest.approx(rho)
    assert a_zcdp_p2.budget_spent == pytest.approx(rho)
    assert a_zcdp.budget_spent == pytest.approx(rho * 3)
    assert a0.budget_spent == pytest.approx((eps_expected(rho * 3, delta), delta))

    a_zcdp_p1.spend(rho)

    assert a_zcdp_p1.budget_spent == pytest.approx(rho * 2)
    assert a_zcdp_p2.budget_spent == pytest.approx(rho)
    assert a_zcdp.budget_spent == pytest.approx(rho * 4)
    assert a0.budget_spent == pytest.approx((eps_expected(rho * 4, delta), delta))

    with pytest.raises(BudgetExceededError):
        a_zcdp_p1.spend(0.1)

    # exceeding budget should not be counted
    assert a_zcdp_p1.budget_spent == pytest.approx(rho * 2)
    assert a_zcdp_p2.budget_spent == pytest.approx(rho)
    assert a_zcdp.budget_spent == pytest.approx(rho * 4)
    assert a0.budget_spent == pytest.approx((eps_expected(rho * 4, delta), delta))

    with pytest.raises(Exception):
        a_zcdp.spend(-1.0)

def test_rdp_accountant() -> None:
    delta = 1e-6
    alpha = [2.0, 4.0, 8.0, 16.0]

    a0 = ApproxDPAccountant(budget_limit=(10.0, delta))
    a0.set_as_root(name=str(uuid.uuid4()))

    assert a0.budget_spent == (0, 0)

    # should raise error without delta
    with pytest.raises(Exception):
        RDPAccountant(alpha=alpha, parent=a0)

    a_rdp = RDPAccountant(alpha=alpha, parent=a0, delta=delta)

    assert a_rdp.budget_spent == {alpha: 0.0 for alpha in alpha}
    assert a_rdp.alpha == alpha

    # After creating RDPAccountant, delta should be spent
    assert a0.budget_spent == pytest.approx((0, delta))

    # RDP to (ε,δ)-DP conversion (Mironov 2017, Proposition 3)
    def rdp_to_eps(rdp_budget: dict[float, float], delta: float) -> float:
        return min(eps + math.log(1 / delta) / (alpha - 1)
                   for alpha, eps in rdp_budget.items())

    # Spend budget with eps for each alpha
    budget1 = {alpha: 0.1 for alpha in alpha}
    a_rdp.spend(budget1)

    assert a_rdp.budget_spent == budget1
    expected_eps = rdp_to_eps(budget1, delta)
    assert a0.budget_spent[0] == pytest.approx(expected_eps)
    assert a0.budget_spent[1] == pytest.approx(delta)

    # Spend more budget
    budget2 = {alpha: 0.05 for alpha in alpha}
    a_rdp.spend(budget2)

    combined = {alpha: 0.15 for alpha in alpha}
    assert a_rdp.budget_spent == pytest.approx(combined)
    expected_eps = rdp_to_eps(combined, delta)
    assert a0.budget_spent[0] == pytest.approx(expected_eps)

    # Parallel composition
    a_rdp_p = RDPParallelAccountant(parent=a_rdp)
    a_rdp_p1 = RDPAccountant(alpha=alpha, parent=a_rdp_p)
    a_rdp_p2 = RDPAccountant(alpha=alpha, parent=a_rdp_p)

    budget3 = {alpha: 0.02 for alpha in alpha}
    a_rdp_p1.spend(budget3)
    a_rdp_p2.spend(budget3)

    assert a_rdp_p1.budget_spent == pytest.approx(budget3)
    assert a_rdp_p2.budget_spent == pytest.approx(budget3)
    # Parallel composition takes max, so a_rdp should have 0.15 + 0.02 = 0.17
    expected_rdp = {alpha: 0.17 for alpha in alpha}
    assert a_rdp.budget_spent == pytest.approx(expected_rdp)

    # One branch spends more
    budget4 = {alpha: 0.03 for alpha in alpha}
    a_rdp_p1.spend(budget4)

    assert a_rdp_p1.budget_spent == pytest.approx({alpha: 0.05 for alpha in alpha})
    assert a_rdp_p2.budget_spent == pytest.approx(budget3)
    # Parallel: max(0.05, 0.02) = 0.05, so a_rdp should have 0.15 + 0.05 = 0.20
    expected_rdp = {alpha: 0.20 for alpha in alpha}
    assert a_rdp.budget_spent == pytest.approx(expected_rdp)

    # Budget limit exceeded
    a0_new = ApproxDPAccountant(budget_limit=(10.0, 1e-3))
    a0_new.set_as_root(name=str(uuid.uuid4()))
    a_rdp_limited = RDPAccountant(
        alpha=alpha,
        budget_limit={alpha: 0.1 for alpha in alpha},
        parent=a0_new,
        delta=delta,
    )
    a_rdp_limited.spend({alpha: 0.05 for alpha in alpha})
    with pytest.raises(BudgetExceededError):
        a_rdp_limited.spend({alpha: 0.1 for alpha in alpha})

    # Invalid alpha (must be > 1)
    with pytest.raises(ValueError):
        RDPAccountant(alpha=[0.5, 2.0], parent=a0, delta=delta)

    with pytest.raises(ValueError):
        RDPAccountant(alpha=[1.0, 2.0], parent=a0, delta=delta)

    # Empty alpha should raise error
    with pytest.raises(ValueError):
        RDPAccountant(alpha=[], parent=a0, delta=delta)

def test_rdp_subsampling_accountant() -> None:
    delta = 1e-6
    alpha = [2.0, 4.0, 8.0, 16.0]  # Must be integers >= 2 for subsampled RDP

    a0 = ApproxDPAccountant(budget_limit=(10.0, delta))
    a0.set_as_root(name=str(uuid.uuid4()))

    a_rdp = RDPAccountant(alpha=alpha, parent=a0, delta=delta)

    # Create subsampling accountant
    a_sub = a_rdp.create_subsampling_accountant(sampling_rate=0.01)

    assert isinstance(a_sub, RDPAccountant)
    assert a_sub.budget_spent == {a: 0.0 for a in alpha}

    # Spend budget (simulating subsampled mechanism)
    budget1 = {a: 0.001 for a in alpha}
    a_sub.spend(budget1)

    assert a_sub.budget_spent == pytest.approx(budget1)
    assert a_rdp.budget_spent == pytest.approx(budget1)

    # Second spend should fail (single-use restriction)
    with pytest.raises(DPError):
        a_sub.spend(budget1)
