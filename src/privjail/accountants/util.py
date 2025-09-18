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
from typing import Generic, TypeVar, Any, Sequence

class BudgetExceededError(Exception):
    pass

T = TypeVar("T")

class Accountant(ABC, Generic[T]):
    _budget_limit : T | None
    _budget_spent : T
    _parent       : Accountant[Any] | None
    _root_name    : str
    _depth        : int

    def __init__(self,
                 *,
                 budget_limit : T | None               = None,
                 parent       : Accountant[Any] | None = None,
                 ):
        if budget_limit is not None:
            type(self).assert_budget(budget_limit)

        self._budget_limit = budget_limit
        self._budget_spent = type(self).zero()
        self._parent       = parent
        self._root_name    = parent._root_name if parent is not None else ""
        self._depth        = parent._depth + 1 if parent is not None else 0

    def set_as_root(self, name: str) -> None:
        assert self._root_name == ""
        self._root_name = name

        register_root_accountant(name, self)

    def budget_spent(self) -> T:
        return self._budget_spent

    def get_parent(self) -> Accountant[Any]:
        if self._parent is None:
            raise RuntimeError
        return self._parent

    def spend(self, budget: T) -> None:
        type(self).assert_budget(budget)

        next_budget_spent = type(self).compose(self._budget_spent, budget)

        if self._budget_limit is not None and type(self).exceeds(next_budget_spent, self._budget_limit):
            raise BudgetExceededError()

        if self._parent is not None:
            self.propagate(next_budget_spent, self._parent)

        self._budget_spent = next_budget_spent

    @staticmethod
    @abstractmethod
    def family_name() -> str:
        pass

    @abstractmethod
    def propagate(self, next_budget_spent: T, parent: Accountant[Any]) -> None:
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

def get_lsca_of_same_family(accountants: Sequence[Accountant[Any]]) -> Accountant[Any]:
    assert len(accountants) > 0

    family = type(accountants[0]).family_name()
    max_depth = max([a._depth for a in accountants])
    accountants_set = set(accountants)

    for d in reversed(range(max_depth + 1)):
        if any(family != type(a).family_name() for a in accountants_set):
            raise RuntimeError("Encountered accountants of different family before reaching LSCA")

        if len(accountants_set) == 1:
            return next(iter(accountants_set))

        accountants_set = {a.get_parent() if a._depth == d else a for a in accountants_set}

    raise RuntimeError

root_accountants : dict[str, Accountant[Any]] = {}

def register_root_accountant(name: str, accountant: Accountant[Any]) -> None:
    global root_accountants

    if name in root_accountants:
        raise ValueError(f"Name '{name}' has alreadly been registered")

    root_accountants[name] = accountant

def get_root_accountant(name: str) -> Accountant[Any]:
    global root_accountants

    if name not in root_accountants:
        raise ValueError(f"Name '{name}' does not exist")

    return root_accountants[name]

def get_all_root_accountants() -> dict[str, Accountant[Any]]:
    global root_accountants
    return root_accountants

# should not be exposed
def clear_root_accountants() -> None:
    global root_accountants
    root_accountants = {}
