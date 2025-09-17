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

PureBudgetType = float

class PureAccountant(Accountant[float]):
    @staticmethod
    def family_name() -> str:
        return "pure"

    @staticmethod
    def propagate(budget_spent: float, next_budget_spent: float, parent: Accountant[Any]) -> None:
        if isinstance(parent, PureParallelAccountant):
            parent.spend(next_budget_spent)
        else:
            raise Exception

    @staticmethod
    def compose(budget1: float, budget2: float) -> float:
        return budget1 + budget2

    @staticmethod
    def zero() -> float:
        return 0.0

    @staticmethod
    def exceeds(budget1: float, budget2: float) -> bool:
        return budget1 > budget2

    @staticmethod
    def assert_budget(budget: float) -> None:
        assert budget >= 0

    @staticmethod
    def parallel_accountant() -> type[Accountant[float]]:
        return PureParallelAccountant

class PureParallelAccountant(ParallelAccountant[float]):
    @staticmethod
    def family_name() -> str:
        return "pure"

    @staticmethod
    def propagate(budget_spent: float, next_budget_spent: float, parent: Accountant[Any]) -> None:
        if isinstance(parent, PureAccountant):
            parent.spend(next_budget_spent - budget_spent)
        else:
            raise Exception

    @staticmethod
    def compose(budget1: float, budget2: float) -> float:
        return max(budget1, budget2)

    @staticmethod
    def zero() -> float:
        return 0.0

    @staticmethod
    def exceeds(budget1: float, budget2: float) -> bool:
        return budget1 > budget2

    @staticmethod
    def assert_budget(budget: float) -> None:
        assert budget >= 0
