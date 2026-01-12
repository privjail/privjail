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

from .util import Accountant, ParallelAccountant
from .. import egrpc

ApproxBudgetType = tuple[float, float]

@egrpc.remoteclass
class ApproxAccountant(Accountant[ApproxBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "approx"

    def propagate(self, next_budget_spent: ApproxBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, ApproxParallelAccountant):
            parent.spend(next_budget_spent)
        else:
            raise Exception

    @staticmethod
    def compose(budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> ApproxBudgetType:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return (eps1 + eps2, delta1 + delta2)

    @staticmethod
    def zero() -> ApproxBudgetType:
        return (0.0, 0.0)

    @staticmethod
    def exceeds(budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> bool:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return eps1 > eps2 or delta1 > delta2

    @staticmethod
    def assert_budget(budget: ApproxBudgetType) -> None:
        eps, delta = budget
        assert eps >= 0 and delta >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> ApproxBudgetType | None:
        if budget is None:
            return None
        elif isinstance(budget, tuple) and len(budget) == 2:
            eps, delta = budget
            normalized = (float(eps), float(delta))
            cls.assert_budget(normalized)
            return normalized
        else:
            raise TypeError("Approx accountant budget must be a tuple of two float values.")

    @staticmethod
    def parallel_accountant() -> type[Accountant[ApproxBudgetType]]:
        return ApproxParallelAccountant

class ApproxParallelAccountant(ParallelAccountant[ApproxBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "approx"

    def propagate(self, next_budget_spent: ApproxBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, ApproxAccountant):
            eps, delta = self._budget_spent
            next_eps, next_delta = next_budget_spent
            parent.spend((next_eps - eps, next_delta - delta))
        else:
            raise Exception

    @staticmethod
    def compose(budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> ApproxBudgetType:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return (max(eps1, eps2), max(delta1, delta2))

    @staticmethod
    def zero() -> ApproxBudgetType:
        return (0.0, 0.0)

    @staticmethod
    def exceeds(budget1: ApproxBudgetType, budget2: ApproxBudgetType) -> bool:
        eps1, delta1 = budget1
        eps2, delta2 = budget2
        return eps1 > eps2 or delta1 > delta2

    @staticmethod
    def assert_budget(budget: ApproxBudgetType) -> None:
        eps, delta = budget
        assert eps >= 0 and delta >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> ApproxBudgetType | None:
        return ApproxAccountant.normalize_budget(budget)
