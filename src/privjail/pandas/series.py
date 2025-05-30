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
from typing import overload, TypeVar, Any, Literal, Generic, Sequence
import warnings
import copy

import numpy as _np
import pandas as _pd

from ..util import DPError, is_realnum, realnum, floating
from ..prisoner import SensitiveInt, SensitiveFloat, _max as smax, _min as smin
from ..distance import Distance
from .. import egrpc
from .util import ElementType, PrivPandasBase, PrivPandasExclusiveDummy, assert_ptag, total_max_distance
from .domain import Domain, BoolDomain, RealDomain, CategoryDomain

T = TypeVar("T")

# to avoid TypeError: type 'Series' is not subscriptable
# class PrivSeries(PrivPandasBase[_pd.Series[T]]):
@egrpc.remoteclass
class PrivSeries(Generic[T], PrivPandasBase[_pd.Series]): # type: ignore[type-arg]
    """Private Series.

    Each value in this series object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of values is treated as a sensitive value.
    """
    _domain : Domain

    def __init__(self,
                 data         : Any,
                 domain       : Domain,
                 distance     : Distance,
                 *,
                 parents      : Sequence[PrivPandasBase[Any]] = [],
                 root_name    : str | None                    = None,
                 preserve_row : bool | None                   = None,
                 ):
        self._domain = domain
        ser = _pd.Series(data)
        super().__init__(value=ser, distance=distance, parents=parents, root_name=root_name, preserve_row=preserve_row)

    def _get_dummy_ser(self, n_rows: int = 3) -> _pd.Series[str]:
        index = list(range(n_rows)) + ['...']
        dummy_data = ['***' for _ in range(n_rows)] + ['...']
        # TODO: dtype becomes 'object'
        return _pd.Series(dummy_data, index=index, name=self.name)

    def __str__(self) -> str:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_ser().__str__().replace("dtype: object", f"dtype: {self.domain.dtype}")

    def __repr__(self) -> str:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_ser().__repr__().replace("dtype: object", f"dtype: {self.domain.dtype}")

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(ser) is not supported. Use ser.shape[0] or ser.size instead.")

    @egrpc.multimethod
    def __getitem__(self, key: PrivSeries[bool]) -> PrivSeries[T]:
        assert_ptag(self, key)
        return PrivSeries[T](data         = self._value.__getitem__(key._value),
                             domain       = self.domain,
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = False)

    @egrpc.multimethod
    def __setitem__(self, key: PrivSeries[bool], value: ElementType) -> None:
        assert_ptag(self, key)
        self._value[key._value] = value

    @egrpc.multimethod
    def __eq__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]:
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value == other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__eq__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value == other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __ne__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]:
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value != other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__ne__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value != other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __lt__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]: # type: ignore
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value < other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__lt__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value < other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __le__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]: # type: ignore
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value <= other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__le__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value <= other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __gt__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]: # type: ignore
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value > other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__gt__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value > other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __ge__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]: # type: ignore
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value >= other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__ge__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value >= other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __and__(self, other: PrivSeries[bool]) -> PrivSeries[bool]:
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value & other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__and__.register
    def _(self, other: bool) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value & other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __rand__(self, other: PrivSeries[bool]) -> PrivSeries[bool]: # type: ignore
        assert_ptag(self, other)
        return PrivSeries[bool](data         = other._value & self._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__rand__.register
    def _(self, other: bool) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = other & self._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __or__(self, other: PrivSeries[bool]) -> PrivSeries[bool]:
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value | other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__or__.register
    def _(self, other: bool) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value | other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __ror__(self, other: PrivSeries[bool]) -> PrivSeries[bool]: # type: ignore
        assert_ptag(self, other)
        return PrivSeries[bool](data         = other._value | self._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__ror__.register
    def _(self, other: bool) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = other | self._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __xor__(self, other: PrivSeries[bool]) -> PrivSeries[bool]:
        assert_ptag(self, other)
        return PrivSeries[bool](data         = self._value ^ other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__xor__.register
    def _(self, other: bool) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value ^ other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __rxor__(self, other: PrivSeries[bool]) -> PrivSeries[bool]: # type: ignore
        assert_ptag(self, other)
        return PrivSeries[bool](data         = other._value ^ self._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__rxor__.register
    def _(self, other: bool) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = other ^ self._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.method
    def __invert__(self) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = ~self._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.property
    def max_distance(self) -> realnum:
        return self.distance.max()

    @egrpc.property
    def shape(self) -> tuple[SensitiveInt]:
        nrows = SensitiveInt(value=self._value.shape[0], distance=self.distance, parents=[self])
        return (nrows,)

    @egrpc.property
    def size(self) -> SensitiveInt:
        return SensitiveInt(value=self._value.size, distance=self.distance, parents=[self])

    # FIXME
    @property
    def dtype(self) -> Any:
        return self._value.dtype

    @egrpc.property
    def name(self) -> str | None:
        return str(self._value.name) if self._value.name is not None else None

    @egrpc.property
    def domain(self) -> Domain:
        return self._domain

    # TODO: add test
    @egrpc.method
    def head(self, n: int = 5) -> PrivSeries[T]:
        return PrivSeries[T](data         = self._value.head(n),
                             domain       = self.domain,
                             distance     = self.distance * 2,
                             parents      = [self],
                             preserve_row = False)

    # TODO: add test
    @egrpc.method
    def tail(self, n: int = 5) -> PrivSeries[T]:
        return PrivSeries[T](data         = self._value.tail(n),
                             domain       = self.domain,
                             distance     = self.distance * 2,
                             parents      = [self],
                             preserve_row = False)

    @overload
    def sort_values(self,
                    *,
                    ascending : bool = ...,
                    inplace   : Literal[True],
                    ) -> None: ...

    @overload
    def sort_values(self,
                    *,
                    ascending : bool = ...,
                    inplace   : Literal[False] = ...,
                    ) -> PrivSeries[T]: ...

    # TODO: add test
    @egrpc.method
    def sort_values(self,
                    *,
                    ascending : bool = True,
                    inplace   : bool = False,
                    ) -> PrivSeries[T] | None:
        if inplace:
            self._value.sort_values(ascending=ascending, inplace=inplace, kind="stable")
            self.renew_ptag()
            return None
        else:
            return PrivSeries[T](data         = self._value.sort_values(ascending=ascending, inplace=inplace, kind="stable"),
                                 domain       = self.domain,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = False)

    @overload
    def replace(self,
                to_replace : ElementType | None = ...,
                value      : ElementType | None = ...,
                *,
                inplace    : Literal[True],
                ) -> None: ...

    @overload
    def replace(self,
                to_replace : ElementType | None = ...,
                value      : ElementType | None = ...,
                *,
                inplace    : Literal[False] = ...,
                ) -> PrivSeries[T]: ...

    @egrpc.method
    def replace(self,
                to_replace : ElementType | None = None,
                value      : ElementType | None = None,
                *,
                inplace    : bool = False,
                ) -> PrivSeries[T] | None:
        if (not is_realnum(to_replace)) or (not is_realnum(value)):
            # TODO: consider string and category dtype
            raise NotImplementedError

        if self.domain.dtype == "int64" and _np.isnan(value):
            new_domain = copy.copy(self.domain)
            new_domain.dtype = "Int64"

        elif isinstance(self.domain, RealDomain):
            a, b = self.domain.range
            if (a is None or a <= to_replace) and (b is None or to_replace <= b):
                new_a = min(a, value) if a is not None else None # type: ignore[type-var]
                new_b = max(b, value) if b is not None else None # type: ignore[type-var]

                new_domain = copy.copy(self.domain)
                new_domain.range = (new_a, new_b)

            else:
                new_domain = self.domain

        else:
            new_domain = self.domain

        if inplace:
            self._value.replace(to_replace, value, inplace=inplace) # type: ignore[arg-type]
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data         = self._value.replace(to_replace, value, inplace=inplace), # type: ignore[arg-type]
                                 domain       = new_domain,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = True)

    @overload
    def dropna(self,
               *,
               inplace      : Literal[True],
               ignore_index : bool = ...,
               ) -> None: ...

    @overload
    def dropna(self,
               *,
               inplace      : Literal[False] = ...,
               ignore_index : bool = ...,
               ) -> PrivSeries[T]: ...

    @egrpc.method
    def dropna(self,
               *,
               inplace      : bool = False,
               ignore_index : bool = False,
               ) -> PrivSeries[T] | None:
        if ignore_index:
            raise DPError("`ignore_index` must be False. Index cannot be reindexed with positions.")

        if self.domain.dtype == "Int64":
            new_domain = copy.copy(self.domain)
            new_domain.dtype = "int64"
        else:
            new_domain = self.domain

        if inplace:
            self._value.dropna(inplace=inplace)
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data         = self._value.dropna(inplace=inplace),
                                 domain       = new_domain,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = True)

    @overload
    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : Literal[True],
             ) -> None: ...

    @overload
    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : Literal[False] = ...,
             ) -> PrivSeries[T]: ...

    @egrpc.method
    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : bool = False,
             ) -> PrivSeries[T] | None:
        if not isinstance(self.domain, RealDomain):
            raise TypeError("Domain must be real numbers.")

        new_domain = copy.copy(self.domain)
        a, b = self.domain.range
        new_a = a if lower is None else lower if a is None else max(a, lower) # type: ignore[type-var]
        new_b = b if upper is None else upper if b is None else min(b, upper) # type: ignore[type-var]
        new_domain.range = (new_a, new_b)

        if inplace:
            self._value.clip(lower, upper, inplace=inplace) # type: ignore[arg-type]
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data         = self._value.clip(lower, upper, inplace=inplace), # type: ignore[arg-type]
                                 domain       = new_domain,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = True)

    @egrpc.method
    def sum(self) -> SensitiveInt | SensitiveFloat:
        if not isinstance(self.domain, RealDomain):
            raise TypeError("Domain must be real numbers.")

        if None in self.domain.range:
            raise DPError("The range is unbounded. Use clip().")

        a, b = self.domain.range

        if a is None or b is None:
            raise DPError("The range is unbounded. Use clip().")

        new_distance = self.distance * max(_np.abs(a), _np.abs(b))

        s = self._value.sum()

        if self.domain.dtype in ["int64", "Int64"]:
            return SensitiveInt(s, new_distance, parents=[self])
        elif self.domain.dtype in ["float64", "Float64"]:
            return SensitiveFloat(s, new_distance, parents=[self])
        else:
            raise ValueError

    @egrpc.method
    def mean(self, eps: float) -> float:
        if not isinstance(self.domain, RealDomain):
            raise TypeError("Domain must be real numbers.")

        if eps <= 0:
            raise DPError(f"Invalid epsilon ({eps})")

        a, b = self.domain.range

        if a is None or b is None:
            raise DPError("The range is unbounded. Use clip().")

        sum_sensitivity = (self.distance * max(_np.abs(a), _np.abs(b))).max()
        count_sensitivity = self.distance.max()

        self.consume_privacy_budget(eps)

        s = _np.random.laplace(loc=float(self._value.sum()), scale=float(sum_sensitivity) / (eps / 2))
        c = _np.random.laplace(loc=float(self._value.shape[0]), scale=float(count_sensitivity) / (eps / 2))

        return s / c

    def value_counts(self,
                     normalize : bool                     = False,
                     sort      : bool                     = True,
                     ascending : bool                     = False,
                     bins      : int | None               = None,
                     dropna    : bool                     = True,
                     values    : list[ElementType] | None = None, # extra argument for privjail
                     ) -> SensitiveSeries[int]:
        # TODO: make SensitiveSeries a dataclass
        result = self._value_counts_impl(normalize, sort, ascending, bins, dropna, values)
        return SensitiveSeries[int](data=list(result.values()), index=list(result.keys()), dtype="object")

    @egrpc.method
    def _value_counts_impl(self,
                           normalize : bool                     = False,
                           sort      : bool                     = True,
                           ascending : bool                     = False,
                           bins      : int | None               = None,
                           dropna    : bool                     = True,
                           values    : list[ElementType] | None = None, # extra argument for privjail
                           ) -> dict[ElementType, SensitiveInt]:
        if normalize:
            # TODO: what is the sensitivity?
            raise NotImplementedError

        if bins is not None:
            # TODO: support continuous values
            raise NotImplementedError

        if sort:
            raise DPError("The `sort` argument must be False.")

        if isinstance(self.domain, CategoryDomain):
            values = self.domain.categories

        if values is None:
            raise DPError("Please provide the `values` argument to prevent privacy leakage.")

        if not dropna and not any(_np.isnan(values)): # type: ignore
            # TODO: consider handling for pd.NA
            warnings.warn("Counts for NaN will be dropped from the result because NaN is not included in `values`", UserWarning)

        counts = self._value.value_counts(normalize, sort, ascending, bins, dropna)

        # Select only the specified values and fill non-existent counts with 0
        counts = counts.reindex(values).fillna(0).astype(int)

        distances = self.distance.create_exclusive_distances(counts.size)

        prisoner_dummy = PrivPandasExclusiveDummy(parents=[self])

        return {k: SensitiveInt(counts.loc[k], distance=distances[i], parents=[prisoner_dummy])
                for i, k in enumerate(counts.index)}

# to avoid TypeError: type 'Series' is not subscriptable
# class SensitiveSeries(_pd.Series[T]):
class SensitiveSeries(Generic[T], _pd.Series): # type: ignore[type-arg]
    """Sensitive Series.

    Each value in this series object is considered a sensitive value.
    The numbers of values are not sensitive.
    This is typically created by counting queries like `PrivSeries.value_counts()`.
    """
    def max_distance(self) -> realnum:
        return total_max_distance(list(self.values))

    def max(self, *args: Any, **kwargs: Any) -> SensitiveInt | SensitiveFloat:
        # TODO: args?
        return smax(self)

    def min(self, *args: Any, **kwargs: Any) -> SensitiveInt | SensitiveFloat:
        # TODO: args?
        return smin(self)

    def reveal(self, eps: floating, mech: str = "laplace") -> float:
        if mech == "laplace":
            from ..mechanism import laplace_mechanism
            result: float = laplace_mechanism(self, eps)
            return result
        else:
            raise ValueError(f"Unknown DP mechanism: '{mech}'")
