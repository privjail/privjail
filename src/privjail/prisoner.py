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
from typing import TypeVar, Generic, Any, overload, Iterable, cast, Sequence, Iterator
import contextlib

import numpy as _np

from .util import integer, floating, realnum, is_integer, is_floating, DPError
from .realexpr import RealExpr, _max as dmax
from .accountants import Accountant, ParallelAccountant, DummyAccountant, PureAccountant, ApproxAccountant, zCDPAccountant, RDPAccountant, get_lsca_of_same_family, BudgetType, PureBudgetType, ApproxBudgetType, zCDPBudgetType, RDPBudgetType, AccountingGroup
from . import egrpc

T = TypeVar("T")

@egrpc.remoteclass
class Prisoner(Generic[T]):
    _value            : T
    distance          : RealExpr
    _accountant       : Accountant[Any]
    _accounting_group : AccountingGroup | None

    def __init__(self,
                 value            : T,
                 distance         : RealExpr,
                 *,
                 parents          : Sequence[Prisoner[Any]] = [], # FIXME: do we really need parents here?
                 accountant       : Accountant[Any] | None  = None,
                 accounting_group : AccountingGroup | None  = None,
                 ):
        self._value   = value
        self.distance = distance

        if distance.is_zero():
            # constant (public value)
            self._accountant       = DummyAccountant()
            self._accounting_group = None

        elif len(parents) == 0:
            # root prisoner
            if accountant is None:
                raise ValueError("accountant must be set for the root prisoner.")
            self._accountant       = accountant
            self._accounting_group = accounting_group if accounting_group is not None else accountant.accounting_group

        elif accountant is not None:
            # accountant is explicitly specified
            if len(parents) != 1:
                raise ValueError("different accountant cannot be set for a prisoner with multiple parents")
            self._accountant       = accountant
            self._accounting_group = accounting_group if accounting_group is not None else accountant.accounting_group

        elif len(parents) == 1:
            # single parent
            self._accountant       = parents[0].accountant
            self._accounting_group = parents[0]._accounting_group

        else:
            # multiple parents
            lsca_accountant        = get_lsca_of_same_family([p.accountant for p in parents if not isinstance(p.accountant, DummyAccountant)])
            self._accountant       = lsca_accountant.get_parent() if isinstance(lsca_accountant, ParallelAccountant) else lsca_accountant
            self._accounting_group = self._accountant.accounting_group

    @egrpc.property
    def accountant(self) -> Accountant[Any]:
        return self._accountant

    @egrpc.method
    def set_accountant(self, accountant: Accountant[Any]) -> None:
        if self._accounting_group is None:
            self._accountant = accountant
            return
        acc_group = accountant.accounting_group
        if acc_group is None or not acc_group.is_ancestor_or_same(self._accounting_group):
            raise DPError("accountant's accounting_group must be an ancestor or the same as prisoner's accounting_group")
        self._accountant = accountant

    @property
    def accounting_group(self) -> AccountingGroup | None:
        return self._accounting_group

    def __str__(self) -> str:
        return "<***>"

    def __repr__(self) -> str:
        return "<***>"

    @egrpc.property
    def max_distance(self) -> realnum:
        return self.distance.max()

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

    def reveal(self,
               *,
               eps   : floating | None = None,
               delta : floating | None = None,
               rho   : floating | None = None,
               scale : floating | None = None,
               mech  : str             = "laplace",
               ) -> float:
        if mech == "laplace":
            from .mechanism import laplace_mechanism
            return laplace_mechanism(self,
                                     eps   = float(eps)   if eps   is not None else None,
                                     scale = float(scale) if scale is not None else None)
        elif mech == "gaussian":
            from .mechanism import gaussian_mechanism
            return gaussian_mechanism(self,
                                      eps   = float(eps)   if eps   is not None else None,
                                      delta = float(delta) if delta is not None else None,
                                      rho   = float(rho)   if rho   is not None else None,
                                      scale = float(scale) if scale is not None else None)
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

    def reveal(self,
               *,
               eps   : floating | None = None,
               delta : floating | None = None,
               rho   : floating | None = None,
               scale : floating | None = None,
               mech  : str             = "laplace",
               ) -> float:
        if mech == "laplace":
            from .mechanism import laplace_mechanism
            return laplace_mechanism(self,
                                     eps   = float(eps)   if eps   is not None else None,
                                     scale = float(scale) if scale is not None else None)
        elif mech == "gaussian":
            from .mechanism import gaussian_mechanism
            return gaussian_mechanism(self,
                                      eps   = float(eps)   if eps   is not None else None,
                                      delta = float(delta) if delta is not None else None,
                                      rho   = float(rho)   if rho   is not None else None,
                                      scale = float(scale) if scale is not None else None)
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
def create_accountant(accountant_type : str,
                      *,
                      parent          : Accountant[Any],
                      budget_limit    : BudgetType | None      = None,
                      delta           : float | None           = None,
                      alpha           : Sequence[float] | None = None,
                      ) -> Accountant[Any]:
    accountant_type_lower = accountant_type.lower()
    if accountant_type_lower == "pure":
        assert budget_limit is None or isinstance(budget_limit, (int, float))
        return PureAccountant(budget_limit=budget_limit, parent=parent)
    elif accountant_type_lower == "approx":
        assert budget_limit is None or isinstance(budget_limit, tuple)
        return ApproxAccountant(budget_limit=budget_limit, parent=parent)
    elif accountant_type_lower == "zcdp":
        assert budget_limit is None or isinstance(budget_limit, (int, float))
        return zCDPAccountant(budget_limit=budget_limit, parent=parent, delta=delta)
    elif accountant_type_lower == "rdp":
        assert budget_limit is None or isinstance(budget_limit, dict)
        assert alpha is not None
        return RDPAccountant(alpha=alpha, budget_limit=budget_limit, parent=parent, delta=delta)
    else:
        raise ValueError(f"Unknown DP type: {accountant_type}. Expected 'pure', 'approx', 'zcdp', or 'rdp'.")

P = TypeVar("P", bound=Prisoner[Any])

@contextlib.contextmanager
def _dp_context(prisoners       : P | Sequence[P],
                accountant_type : str,
                budget_limit    : BudgetType | None      = None,
                delta           : float | None           = None,
                alpha           : Sequence[float] | None = None,
                ) -> Iterator[P | Sequence[P]]:
    if isinstance(prisoners, Prisoner):
        prisoner_list: list[P] = [prisoners]  # type: ignore[list-item]
    else:
        prisoner_list = list(prisoners)

    if len(prisoner_list) == 0:
        raise ValueError("At least one prisoner is required")

    # Check all prisoners have the same accountant
    first_accountant = prisoner_list[0].accountant
    for p in prisoner_list[1:]:
        if p.accountant is not first_accountant:
            raise DPError("All prisoners must have the same accountant")

    old_accountants = [p.accountant for p in prisoner_list]
    new_accountant = create_accountant(accountant_type, parent=first_accountant, budget_limit=budget_limit, delta=delta, alpha=alpha)

    for p in prisoner_list:
        p.set_accountant(new_accountant)
    try:
        yield prisoners  # type: ignore[misc]
    finally:
        for p, old_acc in zip(prisoner_list, old_accountants):
            p.set_accountant(old_acc)

@contextlib.contextmanager
def pureDP(prisoners: P | Sequence[P], budget_limit: PureBudgetType | None = None) -> Iterator[P | Sequence[P]]:
    with _dp_context(prisoners, "pure", budget_limit) as ctx:
        yield ctx

@contextlib.contextmanager
def approxDP(prisoners: P | Sequence[P], budget_limit: ApproxBudgetType | None = None) -> Iterator[P | Sequence[P]]:
    with _dp_context(prisoners, "approx", budget_limit) as ctx:
        yield ctx

@contextlib.contextmanager
def zCDP(prisoners: P | Sequence[P], budget_limit: zCDPBudgetType | None = None, delta: float | None = None) -> Iterator[P | Sequence[P]]:
    with _dp_context(prisoners, "zcdp", budget_limit, delta) as ctx:
        yield ctx

@contextlib.contextmanager
def RDP(prisoners: P | Sequence[P], alpha: Sequence[float], budget_limit: RDPBudgetType | None = None, delta: float | None = None) -> Iterator[P | Sequence[P]]:
    with _dp_context(prisoners, "rdp", budget_limit, delta, alpha) as ctx:
        yield ctx
