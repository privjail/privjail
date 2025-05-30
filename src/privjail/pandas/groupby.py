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
from typing import Any, Iterator, Mapping

import pandas as _pd

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
