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

from .util import Accountant, ParallelAccountant, SubsamplingAccountant
from .. import egrpc

DummyBudgetType = None

@egrpc.remoteclass
class DummyAccountant(Accountant[None]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs, register=False)

    @staticmethod
    def family_name() -> str:
        return "dummy"

    def propagate(self, next_budget_spent: None, parent: Accountant[Any]) -> None:
        pass

    def compose(self, budget1: None, budget2: None) -> None:
        return None

    def zero(self) -> None:
        return None

    def exceeds(self, budget1: None, budget2: None) -> bool:
        return False

    def assert_budget(self, budget: None) -> None:
        pass

    @classmethod
    def normalize_budget(cls, budget: Any) -> None:
        return None

    @staticmethod
    def parallel_accountant() -> type[DummyParallelAccountant]:
        return DummyParallelAccountant

    @staticmethod
    def subsampling_accountant() -> type[SubsamplingAccountant[None]]:
        raise NotImplementedError("Dummy subsampling accountant is not implemented")

class DummyParallelAccountant(ParallelAccountant[None]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs, register=False)

    @staticmethod
    def family_name() -> str:
        return "dummy"

    def propagate(self, next_budget_spent: None, parent: Accountant[Any]) -> None:
        pass

    def compose(self, budget1: None, budget2: None) -> None:
        return None

    def zero(self) -> None:
        return None

    def exceeds(self, budget1: None, budget2: None) -> bool:
        return False

    @classmethod
    def normalize_budget(cls, budget: Any) -> None:
        return None

    def assert_budget(self, budget: None) -> None:
        pass
