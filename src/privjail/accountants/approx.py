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
from typing import Any
import math

from .util import Accountant, ParallelAccountant, SubsamplingAccountant
from .. import egrpc

ApproxBudgetType = tuple[float, float]

@egrpc.remoteclass
class ApproxDPAccountant(Accountant[ApproxBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "approx"

    @egrpc.property
    def budget_spent(self) -> ApproxBudgetType:
        return self._budget_spent

    def propagate(self, next_budget_spent: ApproxBudgetType, parent: Accountant[Any]) -> None:
        diff = (next_budget_spent[0] - self._budget_spent[0], next_budget_spent[1] - self._budget_spent[1])
        if isinstance(parent, ApproxDPAccountant):
            parent.spend(diff)
        elif isinstance(parent, ApproxDPParallelAccountant):
            parent.spend(next_budget_spent)
        elif isinstance(parent, ApproxDPSubsamplingAccountant):
            parent.spend(next_budget_spent)
        else:
            raise Exception

    def compose(self, budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> ApproxBudgetType:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return (eps1 + eps2, delta1 + delta2)

    def zero(self) -> ApproxBudgetType:
        return (0.0, 0.0)

    def exceeds(self, budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> bool:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return eps1 > eps2 or delta1 > delta2

    def assert_budget(self, budget: ApproxBudgetType) -> None:
        eps, delta = budget
        assert eps >= 0 and delta >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> ApproxBudgetType | None:
        if budget is None:
            return None
        elif isinstance(budget, tuple) and len(budget) == 2:
            eps, delta = budget
            normalized = (float(eps), float(delta))
            assert normalized[0] >= 0 and normalized[1] >= 0
            return normalized
        else:
            raise TypeError("Approx accountant budget must be a tuple of two float values.")

    @staticmethod
    def parallel_accountant() -> type[ApproxDPParallelAccountant]:
        return ApproxDPParallelAccountant

    @staticmethod
    def subsampling_accountant() -> type[ApproxDPSubsamplingAccountant]:
        return ApproxDPSubsamplingAccountant

class ApproxDPParallelAccountant(ParallelAccountant[ApproxBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "approx"

    def propagate(self, next_budget_spent: ApproxBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, ApproxDPAccountant):
            eps, delta = self._budget_spent
            next_eps, next_delta = next_budget_spent
            parent.spend((next_eps - eps, next_delta - delta))
        else:
            raise Exception

    def compose(self, budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> ApproxBudgetType:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return (max(eps1, eps2), max(delta1, delta2))

    def zero(self) -> ApproxBudgetType:
        return (0.0, 0.0)

    def exceeds(self, budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> bool:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return eps1 > eps2 or delta1 > delta2

    def assert_budget(self, budget: ApproxBudgetType) -> None:
        eps, delta = budget
        assert eps >= 0 and delta >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> ApproxBudgetType | None:
        return ApproxDPAccountant.normalize_budget(budget)

class ApproxDPSubsamplingAccountant(SubsamplingAccountant[ApproxBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "approx"

    def propagate(self, next_budget_spent: ApproxBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, ApproxDPAccountant):
            prev_eps, prev_delta = self._budget_spent
            next_eps, next_delta = next_budget_spent
            parent.spend((next_eps - prev_eps, next_delta - prev_delta))
        else:
            raise Exception

    def compose(self, budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> ApproxBudgetType:
        q = self._sampling_rate
        eps2, delta2 = budget2
        amp_eps = math.log(1 + q * (math.exp(eps2) - 1))
        amp_delta = q * delta2
        assert budget1[0] <= amp_eps and budget1[1] <= amp_delta
        return (amp_eps, amp_delta)

    def zero(self) -> ApproxBudgetType:
        return (0.0, 0.0)

    def exceeds(self, budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> bool:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return eps1 > eps2 or delta1 > delta2

    def assert_budget(self, budget: ApproxBudgetType) -> None:
        eps, delta = budget
        assert eps >= 0 and delta >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> ApproxBudgetType | None:
        return ApproxDPAccountant.normalize_budget(budget)
