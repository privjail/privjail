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

import pandas as _pd

from .. import egrpc
from ..util import ElementType

@egrpc.dataclass
class Index:
    values : Sequence[ElementType]
    name   : ElementType | None

    def to_pandas(self) -> _pd.Index[Any]:
        return _pd.Index(self.values, name=self.name)

@egrpc.dataclass
class MultiIndex:
    values : Sequence[tuple[ElementType, ...]]
    names  : Sequence[ElementType | None]

    def to_pandas(self) -> _pd.MultiIndex:
        return _pd.MultiIndex.from_tuples(self.values, names=list(self.names))

def pack_pandas_index(index: _pd.Index[Any]) -> Index | MultiIndex:
    if isinstance(index, _pd.MultiIndex):
        return MultiIndex(index.tolist(), list(index.names))
    else:
        return Index(index.tolist(), index.name)
