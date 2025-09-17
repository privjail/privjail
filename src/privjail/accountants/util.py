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
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any

class BudgetExceededError(Exception):
    pass

T = TypeVar("T")

class Accountant(ABC, Generic[T]):
    def __init__(self,
                 *,
                 budget_limit : T | None               = None,
                 parent       : Accountant[Any] | None = None):
        if budget_limit is not None:
            type(self).assert_budget(budget_limit)

        self._budget_limit = budget_limit
        self._budget_spent = type(self).zero()
        self._parent = parent

    def budget_spent(self) -> T:
        return self._budget_spent

    def spend(self, budget: T) -> None:
        type(self).assert_budget(budget)

        next_budget_spent = type(self).compose(self._budget_spent, budget)

        if self._budget_limit is not None and type(self).exceeds(next_budget_spent, self._budget_limit):
            raise BudgetExceededError()

        if self._parent is not None:
            type(self).propagate(self._budget_spent, next_budget_spent, self._parent)

        self._budget_spent = next_budget_spent

    @staticmethod
    @abstractmethod
    def propagate(budget_spent: T, next_budget_spent: T, parent: Accountant[Any]) -> None:
        pass

    @staticmethod
    @abstractmethod
    def compose(budget1: T, budget2: T) -> T:
        pass

    @staticmethod
    @abstractmethod
    def zero() -> T:
        pass

    @staticmethod
    @abstractmethod
    def exceeds(budget1: T, budget2: T) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def assert_budget(budget: T) -> None:
        pass

    @staticmethod
    @abstractmethod
    def parallel_accountant() -> type[Accountant[T]]:
        pass

class ParallelAccountant(Generic[T], Accountant[T]):
    @staticmethod
    def parallel_accountant() -> type[Accountant[T]]:
        raise Exception
