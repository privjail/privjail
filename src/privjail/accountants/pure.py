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
from .approx import ApproxAccountant

PureBudgetType = float

class PureAccountant(Accountant[PureBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "pure"

    @staticmethod
    def propagate(budget_spent: PureBudgetType, next_budget_spent: PureBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, PureParallelAccountant):
            parent.spend(next_budget_spent)
        elif isinstance(parent, ApproxAccountant):
            # basic composition
            parent.spend((next_budget_spent - budget_spent, 0))
        else:
            raise Exception

    @staticmethod
    def compose(budget1: PureBudgetType, budget2: PureBudgetType) -> PureBudgetType:
        return budget1 + budget2

    @staticmethod
    def zero() -> PureBudgetType:
        return 0.0

    @staticmethod
    def exceeds(budget1: PureBudgetType, budget2: PureBudgetType) -> bool:
        return budget1 > budget2

    @staticmethod
    def assert_budget(budget: PureBudgetType) -> None:
        assert budget >= 0

    @staticmethod
    def parallel_accountant() -> type[Accountant[PureBudgetType]]:
        return PureParallelAccountant

class PureParallelAccountant(ParallelAccountant[PureBudgetType]):
    @staticmethod
    def family_name() -> str:
        return "pure"

    @staticmethod
    def propagate(budget_spent: PureBudgetType, next_budget_spent: PureBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, PureAccountant):
            parent.spend(next_budget_spent - budget_spent)
        else:
            raise Exception

    @staticmethod
    def compose(budget1: PureBudgetType, budget2: PureBudgetType) -> PureBudgetType:
        return max(budget1, budget2)

    @staticmethod
    def zero() -> PureBudgetType:
        return 0.0

    @staticmethod
    def exceeds(budget1: PureBudgetType, budget2: PureBudgetType) -> bool:
        return budget1 > budget2

    @staticmethod
    def assert_budget(budget: PureBudgetType) -> None:
        assert budget >= 0
