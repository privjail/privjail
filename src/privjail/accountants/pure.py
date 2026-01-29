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
from .approx import ApproxDPAccountant
from .. import egrpc

PureBudgetType = float

@egrpc.remoteclass
class PureDPAccountant(Accountant[PureBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "pure"

    @egrpc.property
    def budget_spent(self) -> PureBudgetType:
        return self._budget_spent

    def propagate(self, next_budget_spent: PureBudgetType, parent: Accountant[Any]) -> None:
        diff = next_budget_spent - self._budget_spent
        if isinstance(parent, PureDPAccountant):
            parent.spend(diff)
        elif isinstance(parent, PureDPParallelAccountant):
            parent.spend(next_budget_spent)
        elif isinstance(parent, PureDPSubsamplingAccountant):
            parent.spend(next_budget_spent)
        elif isinstance(parent, ApproxDPAccountant):
            parent.spend((diff, 0))
        else:
            raise Exception

    def compose(self, budget1: PureBudgetType, budget2: PureBudgetType) -> PureBudgetType:
        return budget1 + budget2

    def zero(self) -> PureBudgetType:
        return 0.0

    def exceeds(self, budget1: PureBudgetType, budget2: PureBudgetType) -> bool:
        return budget1 > budget2

    def assert_budget(self, budget: PureBudgetType) -> None:
        assert budget >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> PureBudgetType | None:
        if budget is None:
            return None
        elif isinstance(budget, (int, float)):
            normalized = float(budget)
            assert normalized >= 0
            return normalized
        else:
            raise TypeError("Pure accountant budget must be a single float value.")

    @staticmethod
    def parallel_accountant() -> type[PureDPParallelAccountant]:
        return PureDPParallelAccountant

    @staticmethod
    def subsampling_accountant() -> type[PureDPSubsamplingAccountant]:
        return PureDPSubsamplingAccountant

class PureDPParallelAccountant(ParallelAccountant[PureBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "pure"

    def propagate(self, next_budget_spent: PureBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, PureDPAccountant):
            parent.spend(next_budget_spent - self._budget_spent)
        else:
            raise Exception

    def compose(self, budget1: PureBudgetType, budget2: PureBudgetType) -> PureBudgetType:
        return max(budget1, budget2)

    def zero(self) -> PureBudgetType:
        return 0.0

    def exceeds(self, budget1: PureBudgetType, budget2: PureBudgetType) -> bool:
        return budget1 > budget2

    def assert_budget(self, budget: PureBudgetType) -> None:
        assert budget >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> PureBudgetType | None:
        return PureDPAccountant.normalize_budget(budget)

class PureDPSubsamplingAccountant(SubsamplingAccountant[PureBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "pure"

    def propagate(self, next_budget_spent: PureBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, PureDPAccountant):
            parent.spend(next_budget_spent - self._budget_spent)
        else:
            raise Exception

    def compose(self, budget1: PureBudgetType, budget2: PureBudgetType) -> PureBudgetType:
        q = self._sampling_rate
        amp = math.log(1 + q * (math.exp(budget2) - 1))
        assert budget1 <= amp
        return amp

    def zero(self) -> PureBudgetType:
        return 0.0

    def exceeds(self, budget1: PureBudgetType, budget2: PureBudgetType) -> bool:
        return budget1 > budget2

    def assert_budget(self, budget: PureBudgetType) -> None:
        assert budget >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> PureBudgetType | None:
        return PureDPAccountant.normalize_budget(budget)
