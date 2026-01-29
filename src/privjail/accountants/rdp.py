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
from typing import Any, Sequence
import math

from .util import Accountant, ParallelAccountant, SubsamplingAccountant
from .approx import ApproxDPAccountant
from .. import egrpc
from ..util import DPError

# Budget type: {alpha: epsilon} mapping
RDPBudgetType = dict[float, float]

def rdp_to_approx_eps(rdp_budget: RDPBudgetType, delta: float) -> float:
    nonzero = [(a, eps) for a, eps in rdp_budget.items() if eps > 0]
    if not nonzero:
        return 0.0

    # Rényi Differential Privacy
    # https://arxiv.org/pdf/1702.07476
    # Proposition 3: (α, ε)-RDP implies (ε + log(1/δ)/(α-1), δ)-DP
    return min(eps + math.log(1 / delta) / (a - 1) for a, eps in nonzero)

@egrpc.remoteclass
class RDPAccountant(Accountant[RDPBudgetType]):
    _alpha : list[float]
    _delta : float

    def __init__(self,
                 *,
                 alpha        : Sequence[float] | None = None,
                 budget_limit : RDPBudgetType | None   = None,
                 parent       : Accountant[Any] | None = None,
                 delta        : float | None           = None,
                 prepaid      : bool                   = False,
                 ):
        if alpha is None:
            if isinstance(parent, (RDPAccountant, RDPParallelAccountant, RDPSubsamplingAccountant)):
                self._alpha = parent._alpha
            else:
                raise ValueError("alpha must be specified when parent is not RDPAccountant/RDPParallelAccountant/RDPSubsamplingAccountant")
        else:
            if len(alpha) == 0:
                raise ValueError("alpha must not be empty")
            for a in alpha:
                if a <= 1:
                    raise ValueError(f"alpha must be > 1, got {a}")
            if len(set(alpha)) != len(alpha):
                raise ValueError("alpha must be unique")
            self._alpha = list(alpha)

        if isinstance(parent, ApproxDPAccountant):
            if delta is None:
                raise ValueError("delta must be specified when parent is ApproxDPAccountant")
            if not (0 < delta < 1):
                raise ValueError("delta must be in (0, 1)")
            # Spend delta ahead of time (same pattern as zCDPAccountant)
            parent.spend((0, delta))
            self._delta = delta
        else:
            self._delta = 0.0  # Not used when parent is not ApproxDPAccountant

        super().__init__(budget_limit=budget_limit, parent=parent, prepaid=prepaid)

    @egrpc.property
    def alpha(self) -> list[float]:
        return self._alpha

    @staticmethod
    def family_name() -> str:
        return "RDP"

    @egrpc.property
    def budget_spent(self) -> RDPBudgetType:
        return self._budget_spent

    def propagate(self, next_budget_spent: RDPBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, RDPParallelAccountant):
            parent.spend(next_budget_spent)
        elif isinstance(parent, RDPSubsamplingAccountant):
            parent.spend(next_budget_spent)
        elif isinstance(parent, ApproxDPAccountant):
            prev_eps = rdp_to_approx_eps(self._budget_spent, self._delta)
            next_eps = rdp_to_approx_eps(next_budget_spent, self._delta)
            parent.spend((next_eps - prev_eps, 0))
        else:
            raise Exception

    def compose(self, budget1: RDPBudgetType, budget2: RDPBudgetType) -> RDPBudgetType:
        # Rényi Differential Privacy
        # https://arxiv.org/pdf/1702.07476
        # Proposition 1: Composition is additive for each α
        return {a: budget1[a] + budget2[a] for a in self._alpha}

    def zero(self) -> RDPBudgetType:
        return {a: 0.0 for a in self._alpha}

    def exceeds(self, budget1: RDPBudgetType, budget2: RDPBudgetType) -> bool:
        return any(budget1[a] > budget2[a] for a in self._alpha)

    def assert_budget(self, budget: RDPBudgetType) -> None:
        assert set(budget.keys()) == set(self._alpha)
        for eps in budget.values():
            assert eps >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> RDPBudgetType | None:
        if budget is None:
            return None
        elif isinstance(budget, dict):
            normalized = {float(k): float(v) for k, v in budget.items()}
            for eps in normalized.values():
                assert eps >= 0
            return normalized
        else:
            raise TypeError("RDP accountant budget must be a dict[float, float].")

    @staticmethod
    def parallel_accountant() -> type[RDPParallelAccountant]:
        return RDPParallelAccountant

    @staticmethod
    def subsampling_accountant() -> type[SubsamplingAccountant[RDPBudgetType]]:
        return RDPSubsamplingAccountant

class RDPParallelAccountant(ParallelAccountant[RDPBudgetType]):
    _alpha: list[float]

    def __init__(self,
                 *,
                 alpha        : Sequence[float] | None = None,
                 budget_limit : RDPBudgetType | None   = None,
                 parent       : Accountant[Any] | None = None,
                 ):
        if alpha is not None:
            if len(set(alpha)) != len(alpha):
                raise ValueError("alpha must be unique")
            self._alpha = list(alpha)
        elif isinstance(parent, RDPAccountant):
            self._alpha = parent._alpha
        elif isinstance(parent, RDPParallelAccountant):
            self._alpha = parent._alpha
        else:
            raise ValueError("alpha must be specified or parent must be RDPAccountant/RDPParallelAccountant")

        super().__init__(budget_limit=budget_limit, parent=parent)

    @staticmethod
    def family_name() -> str:
        return "RDP"

    def propagate(self, next_budget_spent: RDPBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, RDPAccountant):
            # Parallel composition: propagate the difference
            # Both next_budget_spent and _budget_spent have all alpha
            diff = {a: next_budget_spent[a] - self._budget_spent[a] for a in self._alpha}
            parent.spend(diff)
        else:
            raise Exception

    def compose(self, budget1: RDPBudgetType, budget2: RDPBudgetType) -> RDPBudgetType:
        # Parallel composition: take max for each alpha
        return {a: max(budget1[a], budget2[a]) for a in self._alpha}

    def zero(self) -> RDPBudgetType:
        return {a: 0.0 for a in self._alpha}

    def exceeds(self, budget1: RDPBudgetType, budget2: RDPBudgetType) -> bool:
        return any(budget1[a] > budget2[a] for a in self._alpha)

    def assert_budget(self, budget: RDPBudgetType) -> None:
        assert set(budget.keys()) == set(self._alpha)
        for eps in budget.values():
            assert eps >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> RDPBudgetType | None:
        return RDPAccountant.normalize_budget(budget)

class RDPSubsamplingAccountant(SubsamplingAccountant[RDPBudgetType]):
    """Subsampling accountant for RDP with Gaussian mechanism.

    This accountant enforces single-use: only one spend() call is allowed,
    because the subsampled Gaussian RDP formula assumes one mechanism
    application per subsampling operation.

    The budget passed to spend() should already be the subsampled RDP budget
    (computed by the mechanism using the Mironov formula).
    """
    _alpha : list[float]
    _used  : bool

    def __init__(self,
                 *,
                 alpha         : Sequence[float] | None = None,
                 sampling_rate : float,
                 budget_limit  : RDPBudgetType | None   = None,
                 parent        : Accountant[Any] | None = None,
                 ):
        if alpha is not None:
            if len(set(alpha)) != len(alpha):
                raise ValueError("alpha must be unique")
            self._alpha = list(alpha)
        elif isinstance(parent, RDPAccountant):
            self._alpha = parent._alpha
        elif isinstance(parent, RDPParallelAccountant):
            self._alpha = parent._alpha
        else:
            raise ValueError("alpha must be specified or parent must be RDPAccountant/RDPParallelAccountant")

        self._used = False
        super().__init__(sampling_rate=sampling_rate, budget_limit=budget_limit, parent=parent)

    @staticmethod
    def family_name() -> str:
        return "RDP"

    def spend(self, budget: RDPBudgetType) -> None:
        if self._used:
            raise DPError(
                "RDPSubsamplingAccountant can only be used once. "
                "Subsampled Gaussian RDP assumes one mechanism application per subsampling."
            )
        self._used = True
        super().spend(budget)

    def propagate(self, next_budget_spent: RDPBudgetType, parent: Accountant[Any]) -> None:
        if isinstance(parent, RDPAccountant):
            parent.spend(next_budget_spent)
        else:
            raise Exception

    def compose(self, budget1: RDPBudgetType, budget2: RDPBudgetType) -> RDPBudgetType:
        del budget1
        return budget2

    def zero(self) -> RDPBudgetType:
        return {a: 0.0 for a in self._alpha}

    def exceeds(self, budget1: RDPBudgetType, budget2: RDPBudgetType) -> bool:
        return any(budget1[a] > budget2[a] for a in self._alpha)

    def assert_budget(self, budget: RDPBudgetType) -> None:
        assert set(budget.keys()) == set(self._alpha)
        for eps in budget.values():
            assert eps >= 0

    @classmethod
    def normalize_budget(cls, budget: Any) -> RDPBudgetType | None:
        return RDPAccountant.normalize_budget(budget)
