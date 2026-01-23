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
from typing import Generic, TypeVar, Any, Sequence, ClassVar
import weakref

from .. import egrpc

class BudgetExceededError(Exception):
    ...

class AccountingGroup:
    _parent          : AccountingGroup | None
    _root_accountant : Accountant[Any]

    def __init__(self,
                 root_accountant : Accountant[Any],
                 parent          : AccountingGroup | None = None):
        self._parent          = parent
        self._root_accountant = root_accountant

    @property
    def parent(self) -> AccountingGroup | None:
        return self._parent

    @property
    def root_accountant(self) -> Accountant[Any]:
        return self._root_accountant

    def is_ancestor_or_same(self, other: AccountingGroup) -> bool:
        current: AccountingGroup | None = other
        while current is not None:
            if current is self:
                return True
            current = current._parent
        return False

    def create_child(self, root_accountant: Accountant[Any]) -> AccountingGroup:
        return AccountingGroup(root_accountant=root_accountant, parent=self)

T = TypeVar("T")

@egrpc.remoteclass
class Accountant(ABC, Generic[T]):
    _budget_limit     : T | None
    _budget_spent     : T
    _parent           : Accountant[Any] | None
    _root_name        : str
    _depth            : int
    _accounting_group : AccountingGroup | None

    accountant_registry: ClassVar[weakref.WeakSet[Accountant[Any]]] = weakref.WeakSet()

    def __init__(self,
                 *,
                 budget_limit     : T | None               = None,
                 parent           : Accountant[Any] | None = None,
                 accounting_group : AccountingGroup | None = None,
                 register         : bool                   = True,
                 ):
        if budget_limit is not None:
            self.assert_budget(budget_limit)

        self._budget_limit     = budget_limit
        self._budget_spent     = self.zero()
        self._parent           = parent
        self._root_name        = parent._root_name if parent is not None else ""
        self._depth            = parent._depth + 1 if parent is not None else 0

        if accounting_group is not None:
            self._accounting_group = accounting_group
        elif parent is not None:
            if isinstance(parent, (ParallelAccountant, SubsamplingAccountant)):
                parent_group = parent._accounting_group
                self._accounting_group = parent_group.create_child(root_accountant=self) if parent_group is not None else AccountingGroup(root_accountant=self)
            else:
                self._accounting_group = parent._accounting_group
        else:
            self._accounting_group = AccountingGroup(root_accountant=self)

        if register:
            Accountant.accountant_registry.add(self)

    def set_as_root(self, name: str) -> None:
        assert self._root_name == ""
        if any(acc._root_name == name for acc in Accountant.accountant_registry):
            raise ValueError(f"Name '{name}' has alreadly been registered")
        self._root_name = name

    def budget_spent(self) -> T:
        return self._budget_spent

    @property
    def accounting_group(self) -> AccountingGroup | None:
        return self._accounting_group

    def get_parent(self) -> Accountant[Any]:
        if self._parent is None:
            raise RuntimeError
        return self._parent

    def spend(self, budget: T) -> None:
        self.assert_budget(budget)

        next_budget_spent = self.compose(self._budget_spent, budget)

        if self._budget_limit is not None and self.exceeds(next_budget_spent, self._budget_limit):
            raise BudgetExceededError()

        if self._parent is not None:
            self.propagate(next_budget_spent, self._parent)

        self._budget_spent = next_budget_spent

    def create_parallel_accountants(self, n_children: int) -> list[Accountant[Any]]:
        if n_children < 0:
            raise ValueError("n_children must be non-negative")
        if n_children == 0:
            return []

        parallel_accountant_type = type(self).parallel_accountant()
        parallel_accountant = parallel_accountant_type(parent=self)
        child_type = type(self)
        return [child_type(parent=parallel_accountant) for _ in range(n_children)]

    def create_subsampling_accountant(self, sampling_rate: float) -> Accountant[Any]:
        subsampling_accountant_type = type(self).subsampling_accountant()
        subsampling_accountant = subsampling_accountant_type(sampling_rate=sampling_rate, parent=self)
        child_type = type(self)
        return child_type(parent=subsampling_accountant)

    @staticmethod
    @abstractmethod
    def family_name() -> str:
        ...

    @abstractmethod
    def propagate(self, next_budget_spent: T, parent: Accountant[Any]) -> None:
        ...

    @abstractmethod
    def compose(self, budget1: T, budget2: T) -> T:
        ...

    @abstractmethod
    def zero(self) -> T:
        ...

    @abstractmethod
    def exceeds(self, budget1: T, budget2: T) -> bool:
        ...

    @abstractmethod
    def assert_budget(self, budget: T) -> None:
        ...

    @classmethod
    @abstractmethod
    def normalize_budget(cls, budget: Any) -> T | None:
        ...

    @staticmethod
    @abstractmethod
    def parallel_accountant() -> type[ParallelAccountant[T]]:
        ...

    @staticmethod
    @abstractmethod
    def subsampling_accountant() -> type[SubsamplingAccountant[T]]:
        ...

class ParallelAccountant(Generic[T], Accountant[T]):
    @staticmethod
    def parallel_accountant() -> type[ParallelAccountant[T]]:
        raise Exception

    @staticmethod
    def subsampling_accountant() -> type[SubsamplingAccountant[T]]:
        raise Exception

class SubsamplingAccountant(Generic[T], Accountant[T]):
    _sampling_rate: float

    def __init__(self,
                 *,
                 sampling_rate    : float,
                 budget_limit     : T | None               = None,
                 parent           : Accountant[Any] | None = None,
                 accounting_group : AccountingGroup | None = None,
                 ):
        if not (0.0 < sampling_rate <= 1.0):
            raise ValueError("sampling_rate must be in (0, 1]")
        self._sampling_rate = sampling_rate
        super().__init__(budget_limit=budget_limit, parent=parent, accounting_group=accounting_group)

    @egrpc.property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @staticmethod
    def parallel_accountant() -> type[ParallelAccountant[T]]:
        raise Exception

    @staticmethod
    def subsampling_accountant() -> type[SubsamplingAccountant[T]]:
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
