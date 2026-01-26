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
import gc

from .util import Accountant
from .pure import PureBudgetType
from .approx import ApproxBudgetType
from .zcdp import zCDPBudgetType
from .rdp import RDPBudgetType
from .. import egrpc

BudgetType = PureBudgetType | ApproxBudgetType | zCDPBudgetType | RDPBudgetType

@egrpc.dataclass
class AccountantState:
    name         : str
    budget_limit : BudgetType | None
    budget_spent : BudgetType | None
    prepaid      : bool
    children     : list[AccountantState]

    def __repr__(self) -> str:
        lines: list[str] = []
        self._format(lines, prefix="", is_last=True, is_root=True)
        return "\n".join(lines)

    def _describe(self) -> str:
        label = f"{self.name}[spent={self.budget_spent}"
        if self.budget_limit is not None:
            label += f", limit={self.budget_limit}"
        if self.prepaid:
            label += ", prepaid=True"
        label += "]"
        return label

    def _format(self, lines: list[str], prefix: str, is_last: bool, *, is_root: bool) -> None:
        connector = "" if is_root else ("└─ " if is_last else "├─ ")
        lines.append(f"{prefix}{connector}{self._describe()}")
        next_prefix = prefix if is_root else prefix + ("   " if is_last else "│  ")
        for index, child in enumerate(self.children):
            child._format(lines, next_prefix, index == len(self.children) - 1, is_root=False)

def get_all_root_accountants() -> dict[str, Accountant[Any]]:
    return {acc._root_name: acc for acc in Accountant.accountant_registry
            if acc._parent is None and acc._root_name}

@egrpc.function
def accountant_state() -> dict[str, AccountantState]:
    gc.collect()
    roots = get_all_root_accountants()

    children: dict[Accountant[Any], list[Accountant[Any]]] = {}
    for acc in Accountant.accountant_registry:
        parent = acc._parent
        if parent is not None:
            children.setdefault(parent, []).append(acc)

    def build(node: Accountant[Any]) -> AccountantState:
        return AccountantState(name         = type(node).__name__,
                               budget_limit = node._budget_limit,
                               budget_spent = node._budget_spent,
                               prepaid      = node._prepaid,
                               children     = [build(child) for child in children.get(node, [])])

    return {root_name: build(acc) for root_name, acc in roots.items()}

@egrpc.function
def budgets_spent() -> dict[str, tuple[str, BudgetType]]:
    return {root_name: (type(acc).family_name(), acc.budget_spent())
            for root_name, acc in get_all_root_accountants().items()}
