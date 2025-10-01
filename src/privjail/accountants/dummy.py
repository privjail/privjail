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

DummyBudgetType = None

class DummyAccountant(Accountant[None]):
    @staticmethod
    def family_name() -> str:
        return "dummy"

    def propagate(self, next_budget_spent: None, parent: Accountant[Any]) -> None:
        pass

    @staticmethod
    def compose(budget1: None, budget2: None) -> None:
        return None

    @staticmethod
    def zero() -> None:
        return None

    @staticmethod
    def exceeds(budget1: None, budget2: None) -> bool:
        return False

    @staticmethod
    def assert_budget(budget: None) -> None:
        pass

    @classmethod
    def normalize_budget(cls, budget: Any) -> None:
        return None

    @staticmethod
    def parallel_accountant() -> type[Accountant[None]]:
        return DummyParallelAccountant

class DummyParallelAccountant(ParallelAccountant[None]):
    @staticmethod
    def family_name() -> str:
        return "dummy"

    def propagate(self, next_budget_spent: None, parent: Accountant[Any]) -> None:
        pass

    @staticmethod
    def compose(budget1: None, budget2: None) -> None:
        return None

    @staticmethod
    def zero() -> None:
        return None

    @staticmethod
    def exceeds(budget1: None, budget2: None) -> bool:
        return False

    @classmethod
    def normalize_budget(cls, budget: Any) -> None:
        return None

    @staticmethod
    def assert_budget(budget: None) -> None:
        pass
