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
from typing import Any, Iterator, Mapping, Sequence, TypeVar, overload
import math
import itertools

import pandas as _pd

from .. import egrpc
from ..util import DPError, ElementType
from ..realexpr import RealExpr, _min
from ..accountants import Accountant
from ..prisoner import Prisoner, SensitiveInt
from .domain import CategoryDomain, sum_sensitivity
from .dataframe import PrivDataFrame, SensitiveDataFrame
from .series import SensitiveSeries

T = TypeVar("T")

def _squash_tuple(t: tuple[T, ...]) -> T | tuple[T, ...]:
    return t[0] if len(t) == 1 else t

@egrpc.remoteclass
class PrivDataFrameGroupBy(Prisoner[_pd.core.groupby.DataFrameGroupBy]): # type: ignore[type-arg]
    # TODO: groups are ordered?
    _df                : PrivDataFrame
    _by_columns        : list[str]
    _by_objs           : list[list[ElementType]]
    _variable_exprs    : dict[tuple[ElementType, ...], RealExpr] | None
    _child_accountants : dict[tuple[ElementType, ...], Accountant[Any]] | None

    def __init__(self,
                 obj               : _pd.core.groupby.DataFrameGroupBy, # type: ignore[type-arg]
                 df                : PrivDataFrame,
                 by_columns        : list[str],
                 by_objs           : list[list[ElementType]],
                 variable_exprs    : dict[tuple[ElementType, ...], RealExpr] | None = None,
                 child_accountants : dict[tuple[ElementType, ...], Accountant[Any]] | None = None,
                 ):
        self._df                = df
        self._by_columns        = by_columns
        self._by_objs           = by_objs
        self._variable_exprs    = variable_exprs
        self._child_accountants = child_accountants
        super().__init__(value=obj, distance=df.distance, parents=[df])

    def _get_variable_exprs(self) -> dict[tuple[ElementType, ...], RealExpr]:
        if self._variable_exprs is None:
            n_children = math.prod(len(ks) for ks in self._by_objs)
            assert len(self) == n_children

            if self._df._is_uldp():
                exprs = self._df._user_max_freq.create_exclusive_children(n_children)
            else:
                exprs = self._df.distance.create_exclusive_children(n_children)

            self._variable_exprs = {o: d for o, d in zip(itertools.product(*self._by_objs), exprs)}

        return self._variable_exprs

    def _get_child_accountants(self) -> dict[tuple[ElementType, ...], Accountant[Any]]:
        if self._child_accountants is None:
            n_children = math.prod(len(ks) for ks in self._by_objs)
            assert len(self) == n_children

            accountant_type = type(self._df.accountant)
            parallel_accountant = accountant_type.parallel_accountant()(parent=self._df.accountant)

            self._child_accountants = {o: accountant_type(parent=parallel_accountant) for o in itertools.product(*self._by_objs)}

        return self._child_accountants

    def _get_index(self) -> _pd.Index[Any] | _pd.MultiIndex:
        if len(self._by_columns) == 1:
            return _pd.Index(self._by_objs[0])
        else:
            return _pd.MultiIndex.from_tuples(itertools.product(*self._by_objs), names=self._by_columns)

    def _gen_empty_df(self) -> _pd.DataFrame:
        columns = self._df.columns
        dtypes = self._df.dtypes
        return _pd.DataFrame({c: _pd.Series(dtype=d) for c, d in zip(columns, dtypes)})

    @egrpc.method
    def __len__(self) -> int:
        prod = 1
        for objs in self._by_objs:
            prod *= len(objs)
        return prod

    def __iter__(self) -> Iterator[tuple[ElementType | tuple[ElementType, ...], PrivDataFrame]]:
        return iter(self.groups.items())

    @egrpc.property
    def groups(self) -> dict[ElementType | tuple[ElementType, ...], PrivDataFrame]:
        # set empty groups for absent keys
        groups = {_squash_tuple(o): self._value.get_group(_squash_tuple(o)) if _squash_tuple(o) in self._value.groups else self._gen_empty_df()
                  for o in itertools.product(*self._by_objs)}

        # TODO: update childrens' category domain that is chosen for the groupby key
        return {o: PrivDataFrame(df,
                                 domains       = self._df.domains,
                                 distance      = self._df.distance if self._df._is_uldp() else e,
                                 user_key      = self._df.user_key,
                                 user_max_freq = e if self._df._is_uldp() else RealExpr.INF,
                                 parents       = [self._df],
                                 accountant    = a,
                                 preserve_row  = False)
                for (o, df), e, a in zip(groups.items(), self._get_variable_exprs().values(), self._get_child_accountants().values())}

    @egrpc.method
    def __getitem__(self, key: str | list[str]) -> PrivDataFrameGroupBy:
        if isinstance(key, list):
            keys = key
        else:
            keys = [key]

        # TODO: column order?
        new_df = self._df[self._by_columns + keys]
        return PrivDataFrameGroupBy(self._value[key], new_df, self._by_columns, self._by_objs,
                                    self._variable_exprs, self._child_accountants)

    @egrpc.method
    def get_group(self, obj: ElementType | tuple[ElementType, ...]) -> PrivDataFrame:
        if isinstance(obj, tuple):
            ot = obj
        else:
            ot = (obj,)

        assert ot in self._get_variable_exprs()

        df = self._value.get_group(obj) if obj in self._value.groups else self._gen_empty_df()
        expr = self._get_variable_exprs()[ot]
        return PrivDataFrame(df,
                             domains       = self._df.domains,
                             distance      = self._df.distance if self._df._is_uldp() else expr,
                             user_key      = self._df.user_key,
                             user_max_freq = expr if self._df._is_uldp() else RealExpr.INF,
                             parents       = [self._df],
                             accountant    = self._get_child_accountants()[ot],
                             preserve_row  = False)

    @egrpc.method
    def size(self) -> SensitiveSeries[int]:
        counts = self._value.size()

        # Select only the groupby keys and fill non-existent counts with 0
        counts = counts.reindex(self._get_index()).fillna(0).astype(int)

        distance = self._df.distance * self._df._user_max_freq if self._df._is_uldp() else self._df.distance

        return SensitiveSeries[int](data           = counts,
                                    distance_group = "ser",
                                    distance       = distance,
                                    parents        = [self._df])

    @egrpc.method
    def sum(self) -> SensitiveDataFrame:
        sums = self._value.sum()

        # Select only the groupby keys and fill non-existent counts with 0
        sums = sums.reindex(self._get_index()).fillna(0)

        distance = self._df.distance * self._df._user_max_freq if self._df._is_uldp() else self._df.distance

        distance_per_ser = [distance * sum_sensitivity(domain)
                            for col, domain in self._df.domains.items() if col not in self._by_columns]

        return SensitiveDataFrame(data             = sums,
                                  distance_group   = "ser",
                                  distance_per_ser = distance_per_ser,
                                  parents          = [self._df])

    def mean(self, eps: float) -> _pd.DataFrame:
        df_sum = self.sum()
        ser_size = self.size()

        # TODO: consider as_index
        n_cols = len(df_sum.columns)
        eps_each = eps / (n_cols + 1)

        return df_sum.reveal(eps=eps_each * n_cols).div(ser_size.reveal(eps=eps_each), axis=0)

# @egrpc.remoteclass
# class PrivDataFrameGroupByUser(Prisoner[_pd.core.groupby.DataFrameGroupBy[ByT, _TT]], Generic[ByT, _TT]):
@egrpc.remoteclass
class PrivDataFrameGroupByUser(Prisoner[_pd.core.groupby.DataFrameGroupBy]): # type: ignore[type-arg]
    _df         : PrivDataFrame
    _by_columns : list[str]

    def __init__(self,
                 obj        : _pd.core.groupby.DataFrameGroupBy, # type: ignore[type-arg]
                 df         : PrivDataFrame,
                 by_columns : list[str],
                 ):
        assert df._is_uldp()
        self._df         = df
        self._by_columns = by_columns
        super().__init__(value=obj, distance=df.distance, parents=[df])

    def __len__(self) -> int:
        raise DPError("This operation is not allowed for user-grouped objects.")

    @egrpc.method
    def __getitem__(self, key: str | list[str]) -> PrivDataFrameGroupByUser:
        # TODO: column order?
        return PrivDataFrameGroupByUser(self._value[key], self._df, self._by_columns)

    def __iter__(self) -> Iterator[tuple[Any, PrivDataFrame]]:
        raise DPError("This operation is not allowed for user-grouped objects.")

    def get_group(self, obj: Any) -> PrivDataFrame:
        raise DPError("This operation is not allowed for user-grouped objects.")

    @egrpc.method
    def head(self, n: int = 5) -> PrivDataFrame:
        return PrivDataFrame(data          = self._value.head(n=n),
                             domains       = self._df.domains,
                             distance      = self._df.distance,
                             user_key      = self._df._user_key,
                             user_max_freq = _min(RealExpr(n), self._df._user_max_freq),
                             parents       = [self._df],
                             preserve_row  = False)

@egrpc.function
def _do_group_by(df         : PrivDataFrame,
                 by         : str | list[str],
                 level      : int | None      = None, # TODO: support multiindex?
                 as_index   : bool            = True,
                 sort       : bool            = True,
                 group_keys : bool            = True,
                 observed   : bool            = True,
                 dropna     : bool            = True,
                 ) -> PrivDataFrameGroupBy | PrivDataFrameGroupByUser:
    # TODO: consider extra arguments
    grouped = df._value.groupby(by, observed=observed)

    if isinstance(by, list):
        by_columns = by
    else:
        by_columns = [by]

    if df.user_key in by_columns:
        return PrivDataFrameGroupByUser(grouped, df, by_columns)

    else:
        by_objs = []

        for col in by_columns:
            key_domain = df.domains[col]

            if not isinstance(key_domain, CategoryDomain):
                raise DPError("Groupby columns must be of a categorical type")

            by_objs.append(list(key_domain.categories))

        return PrivDataFrameGroupBy(grouped, df, by_columns, by_objs)
