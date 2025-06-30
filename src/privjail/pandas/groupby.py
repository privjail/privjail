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
from ..prisoner import Prisoner
from .util import ElementType
from .dataframe import PrivDataFrame, SensitiveDataFrame

class PrivDataFrameGroupBy:
    # TODO: groups are ordered?
    groups     : Mapping[ElementType, PrivDataFrame]
    by_columns : list[str]

    def __init__(self, groups: Mapping[ElementType, PrivDataFrame], by_columns: list[str]):
        self.groups     = groups
        self.by_columns = by_columns

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[tuple[Any, PrivDataFrame]]:
        return iter(self.groups.items())

    def __getitem__(self, key: str | list[str]) -> PrivDataFrameGroupBy:
        if isinstance(key, str):
            keys = [key]
        elif isinstance(key, list):
            keys = key
        else:
            raise TypeError

        # TODO: column order?
        new_groups = {k: df[self.by_columns + keys] for k, df in self.groups.items()}
        return PrivDataFrameGroupBy(new_groups, self.by_columns)

    def get_group(self, key: Any) -> PrivDataFrame:
        return self.groups[key]

    def sum(self) -> SensitiveDataFrame:
        data = [df.drop(self.by_columns, axis=1).sum() for key, df in self.groups.items()]
        return SensitiveDataFrame(data, index=self.groups.keys()) # type: ignore

    def mean(self, eps: float) -> _pd.DataFrame:
        data = [df.drop(self.by_columns, axis=1).mean(eps=eps) for key, df in self.groups.items()]
        return _pd.DataFrame(data, index=self.groups.keys()) # type: ignore

# @egrpc.remoteclass
# class PrivDataFrameGroupByUser(Prisoner[_pd.core.groupby.DataFrameGroupBy[ByT, _TT]], Generic[ByT, _TT]):
@egrpc.remoteclass
class PrivDataFrameGroupByUser(Prisoner[_pd.core.groupby.DataFrameGroupBy]): # type: ignore[type-arg]
    df         : PrivDataFrame
    by_columns : list[str]

    def __init__(self,
                 obj        : _pd.core.groupby.DataFrameGroupBy, # type: ignore[type-arg]
                 df         : PrivDataFrame,
                 by_columns : list[str],
                 ):
        assert df._is_uldp()
        self.df         = df
        self.by_columns = by_columns
        super().__init__(value=obj, distance=df.distance, parents=[df])

    def __len__(self) -> int:
        raise DPError("This operation is not allowed for user-grouped objects.")

    @egrpc.method
    def __getitem__(self, key: str | list[str]) -> PrivDataFrameGroupByUser:
        # TODO: column order?
        return PrivDataFrameGroupByUser(self._value[key], self.df, self.by_columns)

    def __iter__(self) -> Iterator[tuple[Any, PrivDataFrame]]:
        raise DPError("This operation is not allowed for user-grouped objects.")

    def get_group(self, key: Any) -> PrivDataFrame:
        raise DPError("This operation is not allowed for user-grouped objects.")

    @egrpc.method
    def head(self, n: int = 5) -> PrivDataFrame:
        return PrivDataFrame(data          = self._value.head(n=n),
                             domains       = self.df.domains,
                             distance      = self.df.distance,
                             user_key      = self.df._user_key,
                             user_max_freq = min(n, self.df._user_max_freq) if self.df._user_max_freq is not None else n,
                             parents       = [self.df],
                             preserve_row  = False)

@egrpc.function
def _group_by_user(df         : PrivDataFrame,
                   by         : str, # TODO: support more
                   level      : int | None                   = None, # TODO: support multiindex?
                   as_index   : bool                         = True,
                   sort       : bool                         = True,
                   group_keys : bool                         = True,
                   observed   : bool                         = True,
                   dropna     : bool                         = True,
                   keys       : Sequence[ElementType] | None = None, # extra argument for privjail
                   ) -> PrivDataFrameGroupByUser:
    if df.user_key != by:
        raise DPError("Something went wrong.")

    # TODO: consider extra arguments
    grouped = df._value.groupby(by, observed=observed)
    return PrivDataFrameGroupByUser(grouped, df, [by])
