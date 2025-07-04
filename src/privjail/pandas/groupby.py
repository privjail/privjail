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
from typing import Any, Iterator, Mapping, Sequence

import pandas as _pd

from .. import egrpc
from ..util import DPError
from ..realexpr import RealExpr, _min
from ..prisoner import Prisoner
from .util import ElementType, PrivPandasExclusiveDummy
from .domain import CategoryDomain
from .dataframe import PrivDataFrame, SensitiveDataFrame

@egrpc.remoteclass
class PrivDataFrameGroupBy(Prisoner[_pd.core.groupby.DataFrameGroupBy]): # type: ignore[type-arg]
    # TODO: groups are ordered?
    _df                       : PrivDataFrame
    _by_columns               : list[str]
    _by_keys                  : list[list[ElementType]]
    _variable_exprs           : dict[ElementType, RealExpr] | None
    _exclusive_prisoner_dummy : PrivPandasExclusiveDummy

    def __init__(self,
                 obj                      : _pd.core.groupby.DataFrameGroupBy, # type: ignore[type-arg]
                 df                       : PrivDataFrame,
                 by_columns               : list[str],
                 by_keys                  : list[list[ElementType]],
                 variable_exprs           : dict[ElementType, RealExpr] | None = None,
                 exclusive_prisoner_dummy : PrivPandasExclusiveDummy | None    = None,
                 ):
        self._df                       = df
        self._by_columns               = by_columns
        self._by_keys                  = by_keys
        self._variable_exprs           = variable_exprs
        self._exclusive_prisoner_dummy = PrivPandasExclusiveDummy(parents=[df]) if exclusive_prisoner_dummy is None else exclusive_prisoner_dummy
        super().__init__(value=obj, distance=df.distance, parents=[df])

    def _get_variable_exprs(self) -> dict[ElementType, RealExpr]:
        if self._variable_exprs is None:
            if self._df._is_uldp():
                exprs = self._df._user_max_freq.create_exclusive_children(len(self))
            else:
                exprs = self._df.distance.create_exclusive_children(len(self))

            # TODO: support groupby multiple columns
            assert len(self._by_keys) == 1
            self._variable_exprs = {key: d for key, d in zip(self._by_keys[0], exprs)}

        return self._variable_exprs

    def _gen_empty_df(self) -> _pd.DataFrame:
        columns = self._df.columns
        dtypes = self._df.dtypes
        return _pd.DataFrame({c: _pd.Series(dtype=d) for c, d in zip(columns, dtypes)})

    @egrpc.method
    def __len__(self) -> int:
        prod = 1
        for keys in self._by_keys:
            prod *= len(keys)
        return prod

    def __iter__(self) -> Iterator[tuple[ElementType, PrivDataFrame]]:
        # TODO: support groupby multiple columns
        return iter(self.groups.items())

    @egrpc.property
    def groups(self) -> dict[ElementType, PrivDataFrame]:
        # set empty groups for absent keys
        groups = {key: self._value.get_group(key) if key in self._value.groups else self._gen_empty_df() \
                  for key in self._by_keys[0]}

        # TODO: update childrens' category domain that is chosen for the groupby key
        return {key: PrivDataFrame(df,
                                   domains       = self._df.domains,
                                   distance      = self._df.distance if self._df._is_uldp() else e,
                                   user_key      = self._df.user_key,
                                   user_max_freq = e if self._df._is_uldp() else RealExpr.INF,
                                   parents       = [self._exclusive_prisoner_dummy],
                                   preserve_row  = False) \
                for (key, df), e in zip(groups.items(), self._get_variable_exprs().values())}

    @egrpc.method
    def __getitem__(self, key: str | list[str]) -> PrivDataFrameGroupBy:
        if isinstance(key, list):
            keys = key
        else:
            keys = [key]

        # TODO: column order?
        new_df = self._df[self._by_columns + keys]
        return PrivDataFrameGroupBy(self._value[key], new_df, self._by_columns, self._by_keys,
                                    self._variable_exprs, self._exclusive_prisoner_dummy)

    def get_group(self, key: ElementType | tuple[ElementType, ...]) -> PrivDataFrame:
        if isinstance(key, tuple):
            keys = list(key)
        else:
            keys = [key]

        return self._get_group_impl(keys)

    @egrpc.method
    def _get_group_impl(self, keys: list[ElementType]) -> PrivDataFrame:
        if len(keys) != len(self._by_columns):
            raise ValueError

        for key, possible_keys in zip(keys, self._by_keys):
            if key not in possible_keys:
                raise ValueError

        # TODO: support groupby multiple columns
        key = keys[0]
        df = self._value.get_group(key) if key in self._value.groups else self._gen_empty_df()
        expr = self._get_variable_exprs()[key]
        return PrivDataFrame(df,
                             domains       = self._df.domains,
                             distance      = self._df.distance if self._df._is_uldp() else expr,
                             user_key      = self._df.user_key,
                             user_max_freq = expr if self._df._is_uldp() else RealExpr.INF,
                             parents       = [self._exclusive_prisoner_dummy],
                             preserve_row  = False)

    def sum(self) -> SensitiveDataFrame:
        # FIXME
        data = [df.drop(self.by_columns, axis=1, errors="ignore").sum() for key, df in self.groups.items()]
        return SensitiveDataFrame(data, index=self.groups.keys()) # type: ignore

    def mean(self, eps: float) -> _pd.DataFrame:
        # FIXME
        data = [df.drop(self.by_columns, axis=1, errors="ignore").mean(eps=eps) for key, df in self.groups.items()]
        return _pd.DataFrame(data, index=self.groups.keys()) # type: ignore

    # FIXME: non-standard API
    @egrpc.property
    def by_columns(self) -> list[str]:
        return self._by_columns

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

    def get_group(self, key: Any) -> PrivDataFrame:
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
                 by         : str, # TODO: support more
                 level      : int | None                   = None, # TODO: support multiindex?
                 as_index   : bool                         = True,
                 sort       : bool                         = True,
                 group_keys : bool                         = True,
                 observed   : bool                         = True,
                 dropna     : bool                         = True,
                 keys       : Sequence[ElementType] | None = None, # extra argument for privjail
                 ) -> PrivDataFrameGroupBy | PrivDataFrameGroupByUser:
    # TODO: consider extra arguments
    grouped = df._value.groupby(by, observed=observed)

    if df.user_key == by:
        return PrivDataFrameGroupByUser(grouped, df, [by])

    else:
        key_domain = df.domains[by]
        if isinstance(key_domain, CategoryDomain):
            keys = key_domain.categories

        if keys is None:
            raise DPError("Please provide the `keys` argument to prevent privacy leakage for non-categorical columns.")

        return PrivDataFrameGroupBy(grouped, df, [by], [list(keys)])
