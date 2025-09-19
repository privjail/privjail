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
from typing import TypeVar, Generic, Any, overload, Iterable, cast, Sequence

import numpy as _np

from .util import integer, floating, realnum, is_integer, is_floating
from .realexpr import RealExpr, _max as dmax
from .accountants import *
from . import egrpc

T = TypeVar("T")

class Prisoner(Generic[T]):
    _value          : T
    distance        : RealExpr
    accountant      : Accountant[Any]
    accountant_orig : Accountant[Any]

    def __init__(self,
                 value      : T,
                 distance   : RealExpr,
                 *,
                 parents    : Sequence[Prisoner[Any]] = [], # FIXME: do we really need parents here?
                 accountant : Accountant[Any] | None  = None,
                 ):
        self._value   = value
        self.distance = distance

        if distance.is_zero():
            # constant (public value)
            self.accountant = DummyAccountant()

        elif len(parents) == 0:
            # root prisoner
            if accountant is None:
                raise ValueError("accountant must be set for the root prisoner.")

            self.accountant = accountant

        elif accountant is not None:
            # accountant is explicitly specified
            if len(parents) != 1:
                raise ValueError("different accountant cannot be set for a prisoner with multiple parents")

            self.accountant = accountant

        elif len(parents) == 1:
            # single parent
            self.accountant = parents[0].accountant

        else:
            # multiple parents
            lsca_accountant = get_lsca_of_same_family([p.accountant for p in parents if not isinstance(p.accountant, DummyAccountant)])
            self.accountant = lsca_accountant.get_parent() if isinstance(lsca_accountant, ParallelAccountant) else lsca_accountant

        self.accountant_orig = self.accountant

    def __str__(self) -> str:
        return "<***>"

    def __repr__(self) -> str:
        return "<***>"

    @egrpc.property
    def max_distance(self) -> realnum:
        return self.distance.max()

    @egrpc.method
    def switch_to_pureDP(self, budget_limit: PureBudgetType | None = None) -> None:
        self.accountant = PureAccountant(budget_limit=budget_limit, parent=self.accountant_orig)

    @egrpc.method
    def switch_to_approxDP(self, budget_limit: ApproxBudgetType | None = None) -> None:
        self.accountant = ApproxAccountant(budget_limit=budget_limit, parent=self.accountant_orig)

    @egrpc.method
    def switch_to_zCDP(self, budget_limit: zCDPBudgetType | None = None, delta: float | None = None) -> None:
        self.accountant = zCDPAccountant(budget_limit=budget_limit, parent=self.accountant_orig, delta=delta)

    @egrpc.method
    def switch_to_original_accountant(self) -> None:
        self.accountant = self.accountant_orig

@egrpc.remoteclass
class SensitiveInt(Prisoner[integer]):
    def __init__(self,
                 value      : integer,
                 distance   : RealExpr                = RealExpr(0),
                 *,
                 parents    : Sequence[Prisoner[Any]] = [],
                 accountant : Accountant[Any] | None  = None,
                 ):
        if not is_integer(value):
            raise ValueError("`value` must be int for SensitveInt.")
        super().__init__(value, distance, parents=parents, accountant=accountant)

    def __str__(self) -> str:
        return "<*** (int)>"

    def __repr__(self) -> str:
        return "<*** (int)>"

    @egrpc.method
    def __neg__(self) -> SensitiveInt:
        return SensitiveInt(-self._value, distance=self.distance, parents=[self])

    @egrpc.multimethod
    def __add__(self, other: integer) -> SensitiveInt:
        return SensitiveInt(self._value + other, distance=self.distance, parents=[self])

    @__add__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(self._value + other, distance=self.distance, parents=[self])

    @__add__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return SensitiveInt(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])

    @__add__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __radd__(self, other: integer) -> SensitiveInt: # type: ignore[misc]
        return SensitiveInt(other + self._value, distance=self.distance, parents=[self])

    @__radd__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(other + self._value, distance=self.distance, parents=[self])

    @__radd__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return SensitiveInt(other._value + self._value, distance=self.distance + other.distance, parents=[self, other])

    @__radd__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(other._value + self._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __sub__(self, other: integer) -> SensitiveInt:
        return SensitiveInt(self._value - other, distance=self.distance, parents=[self])

    @__sub__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(self._value - other, distance=self.distance, parents=[self])

    @__sub__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return SensitiveInt(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])

    @__sub__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __rsub__(self, other: integer) -> SensitiveInt: # type: ignore[misc]
        return SensitiveInt(other - self._value, distance=self.distance, parents=[self])

    @__rsub__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(other - self._value, distance=self.distance, parents=[self])

    @__rsub__.register
    def _(self, other: SensitiveInt) -> SensitiveInt:
        return SensitiveInt(other._value - self._value, distance=self.distance + other.distance, parents=[self, other])

    @__rsub__.register
    def _(self, other: SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(other._value - self._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __mul__(self, other: integer) -> SensitiveInt:
        return SensitiveInt(self._value * other, distance=self.distance * _np.abs(other), parents=[self])

    @__mul__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(self._value * other, distance=self.distance * _np.abs(other), parents=[self])

    @egrpc.multimethod
    def __rmul__(self, other: integer) -> SensitiveInt: # type: ignore[misc]
        return SensitiveInt(other * self._value, distance=self.distance * _np.abs(other), parents=[self])

    @__rmul__.register
    def _(self, other: floating) -> SensitiveFloat:
        return SensitiveFloat(other * self._value, distance=self.distance * _np.abs(other), parents=[self])

    def reveal(self, eps: floating, delta: floating = 0.0, mech: str = "laplace") -> float:
        if mech == "laplace":
            from .mechanism import laplace_mechanism
            result: float = laplace_mechanism(self, eps)
            return result
        elif mech == "gaussian":
            from .mechanism import gaussian_mechanism
            result: float = gaussian_mechanism(self, eps, delta)
            return result
        else:
            raise ValueError(f"Unknown DP mechanism: '{mech}'")

@egrpc.remoteclass
class SensitiveFloat(Prisoner[floating]):
    def __init__(self,
                 value      : floating,
                 distance   : RealExpr                = RealExpr(0),
                 *,
                 parents    : Sequence[Prisoner[Any]] = [],
                 accountant : Accountant[Any] | None  = None,
                 ):
        if not is_floating(value):
            raise ValueError("`value` must be float for SensitveFloat.")
        super().__init__(value, distance, parents=parents, accountant=accountant)

    def __str__(self) -> str:
        return "<*** (float)>"

    def __repr__(self) -> str:
        return "<*** (float)>"

    @egrpc.method
    def __neg__(self) -> SensitiveFloat:
        return SensitiveFloat(-self._value, distance=self.distance, parents=[self])

    @egrpc.multimethod
    def __add__(self, other: realnum) -> SensitiveFloat:
        return SensitiveFloat(self._value + other, distance=self.distance, parents=[self])

    @__add__.register
    def _(self, other: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(self._value + other._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __radd__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        return SensitiveFloat(other + self._value, distance=self.distance, parents=[self])

    @__radd__.register
    def _(self, other: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(other._value + self._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __sub__(self, other: realnum) -> SensitiveFloat:
        return SensitiveFloat(self._value - other, distance=self.distance, parents=[self])

    @__sub__.register
    def _(self, other: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(self._value - other._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __rsub__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        return SensitiveFloat(other - self._value, distance=self.distance, parents=[self])

    @__rsub__.register
    def _(self, other: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
        return SensitiveFloat(other._value - self._value, distance=self.distance + other.distance, parents=[self, other])

    @egrpc.multimethod
    def __mul__(self, other: realnum) -> SensitiveFloat:
        return SensitiveFloat(self._value * other, distance=self.distance * _np.abs(other), parents=[self])

    @egrpc.multimethod
    def __rmul__(self, other: realnum) -> SensitiveFloat: # type: ignore[misc]
        return SensitiveFloat(other * self._value, distance=self.distance * _np.abs(other), parents=[self])

    def reveal(self, eps: floating, delta: floating = 0.0, mech: str = "laplace") -> float:
        if mech == "laplace":
            from .mechanism import laplace_mechanism
            result: float = laplace_mechanism(self, eps)
            return result
        elif mech == "gaussian":
            from .mechanism import gaussian_mechanism
            result: float = gaussian_mechanism(self, eps, delta)
            return result
        else:
            raise ValueError(f"Unknown DP mechanism: '{mech}'")

@egrpc.multifunction
def max2(a: SensitiveInt | SensitiveFloat, b: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
    return SensitiveFloat(max(float(a._value), float(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

@max2.register
def _(a: SensitiveInt, b: SensitiveInt) -> SensitiveInt:
    return SensitiveInt(max(int(a._value), int(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

@overload
def _max(*args: SensitiveInt) -> SensitiveInt: ...
@overload
def _max(*args: SensitiveFloat) -> SensitiveFloat: ...
@overload
def _max(*args: Iterable[SensitiveInt]) -> SensitiveInt: ...
@overload
def _max(*args: Iterable[SensitiveFloat]) -> SensitiveFloat: ...
@overload
def _max(*args: Iterable[SensitiveInt | SensitiveFloat] | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat: ...

def _max(*args: Iterable[SensitiveInt | SensitiveFloat] | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat:
    if len(args) == 0:
        raise TypeError("max() expected at least one argment.")

    if len(args) == 1:
        if not isinstance(args[0], Iterable):
            raise TypeError("The first arg passed to max() is not iterable.")
        iterable = args[0]
    else:
        iterable = cast(tuple[SensitiveInt | SensitiveFloat, ...], args)

    it = iter(iterable)
    try:
        result = next(it)
    except StopIteration:
        raise ValueError("List passed to max() is empty.")

    for x in it:
        result = max2(result, x)

    return result

@egrpc.multifunction
def min2(a: SensitiveInt | SensitiveFloat, b: SensitiveInt | SensitiveFloat) -> SensitiveFloat:
    return SensitiveFloat(min(float(a._value), float(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

@min2.register
def _(a: SensitiveInt, b: SensitiveInt) -> SensitiveInt:
    return SensitiveInt(min(int(a._value), int(b._value)), distance=dmax(a.distance, b.distance), parents=[a, b])

@overload
def _min(*args: SensitiveInt) -> SensitiveInt: ...
@overload
def _min(*args: SensitiveFloat) -> SensitiveFloat: ...
@overload
def _min(*args: Iterable[SensitiveInt]) -> SensitiveInt: ...
@overload
def _min(*args: Iterable[SensitiveFloat]) -> SensitiveFloat: ...
@overload
def _min(*args: Iterable[SensitiveInt | SensitiveFloat] | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat: ...

def _min(*args: Iterable[SensitiveInt | SensitiveFloat] | SensitiveInt | SensitiveFloat) -> SensitiveInt | SensitiveFloat:
    if len(args) == 0:
        raise TypeError("min() expected at least one argment.")

    if len(args) == 1:
        if not isinstance(args[0], Iterable):
            raise TypeError("The first arg passed to min() is not iterable.")
        iterable = args[0]
    else:
        iterable = cast(tuple[SensitiveInt | SensitiveFloat, ...], args)

    it = iter(iterable)
    try:
        result = next(it)
    except StopIteration:
        raise ValueError("List passed to min() is empty.")

    for x in it:
        result = min2(result, x)

    return result

@egrpc.function
def budgets_spent() -> dict[str, tuple[str, BudgetType]]:
    return {name: (type(accountant).family_name(), accountant.budget_spent())
            for name, accountant in get_all_root_accountants().items()}
