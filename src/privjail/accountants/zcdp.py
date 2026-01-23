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
from .approx import ApproxAccountant
from .. import egrpc

zCDPBudgetType = float

@egrpc.remoteclass
class zCDPAccountant(Accountant[zCDPBudgetType]):
    def __init__(self,
                 *,
                 budget_limit : zCDPBudgetType | None  = None,
                 parent       : Accountant[Any] | None = None,
                 delta        : float | None           = None,
                 ):
        if isinstance(parent, ApproxAccountant):
            if delta is None:
                raise ValueError("delta must be specified")

            assert 0 < delta < 1

            # spend delta ahead of time
            # TODO: spend delta lazily at the first time of noise addition
            parent.spend((0, delta))

            self._delta = delta

        super().__init__(budget_limit=budget_limit, parent=parent)

    @staticmethod
    def family_name() -> str:
        return "zCDP"

    def propagate(self, next_budget_spent: zCDPBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, zCDPParallelAccountant):
            parent.spend(next_budget_spent)
        elif isinstance(parent, ApproxAccountant):
            eps = eps_from_rho_delta(self._budget_spent, self._delta)
            next_eps = eps_from_rho_delta(next_budget_spent, self._delta)
            parent.spend((next_eps - eps, 0))
        else:
            raise Exception

    @staticmethod
    def compose(budget1: zCDPBudgetType, budget2: zCDPBudgetType) -> zCDPBudgetType:
        return budget1 + budget2

    @staticmethod
    def zero() -> zCDPBudgetType:
        return 0.0

    @staticmethod
    def exceeds(budget1: zCDPBudgetType, budget2: zCDPBudgetType) -> bool:
        return budget1 > budget2

    @staticmethod
    def assert_budget(budget: zCDPBudgetType) -> None:
        assert budget >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> zCDPBudgetType | None:
        if budget is None:
            return None
        elif isinstance(budget, (int, float)):
            normalized = float(budget)
            cls.assert_budget(normalized)
            return normalized
        else:
            raise TypeError("zCDP accountant budget must be a single float value.")

    @staticmethod
    def parallel_accountant() -> type[zCDPParallelAccountant]:
        return zCDPParallelAccountant

    @staticmethod
    def subsampling_accountant() -> type[SubsamplingAccountant[zCDPBudgetType]]:
        raise NotImplementedError("zCDP subsampling accountant is not implemented")

class zCDPParallelAccountant(ParallelAccountant[zCDPBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "zCDP"

    def propagate(self, next_budget_spent: zCDPBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, zCDPAccountant):
            parent.spend(next_budget_spent - self._budget_spent)
        else:
            raise Exception

    @staticmethod
    def compose(budget1: zCDPBudgetType, budget2: zCDPBudgetType) -> zCDPBudgetType:
        return max(budget1, budget2)

    @staticmethod
    def zero() -> zCDPBudgetType:
        return 0.0

    @staticmethod
    def exceeds(budget1: zCDPBudgetType, budget2: zCDPBudgetType) -> bool:
        return budget1 > budget2

    @staticmethod
    def assert_budget(budget: zCDPBudgetType) -> None:
        assert budget >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> zCDPBudgetType | None:
        return zCDPAccountant.normalize_budget(budget)

def eps_from_rho_delta(rho: float, delta: float) -> float:
    # Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds
    # https://arxiv.org/pdf/1605.02065
    # Proposition 1.3
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))
