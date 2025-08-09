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
from typing import Any, TypeVar, Generic, Sequence

import pandas as _pd

from .. import egrpc
from ..util import DPError, realnum
from ..provenance import ChildrenType
from ..realexpr import RealExpr
from ..prisoner import Prisoner, SensitiveInt, SensitiveFloat

T = TypeVar("T")

ElementType = realnum | str | bool

PTag = int

ptag_count = 0

def new_ptag() -> PTag:
    global ptag_count
    ptag_count += 1
    return ptag_count

class PrivPandasBase(Generic[T], Prisoner[T]):
    _ptag : PTag

    def __init__(self,
                 value         : Any,
                 distance      : RealExpr,
                 parents       : Sequence[PrivPandasBase[Any]],
                 root_name     : str | None,
                 preserve_row  : bool | None,
                 children_type : ChildrenType = "inclusive",
                 ):
        if len(parents) == 0:
            preserve_row = False
        elif preserve_row is None:
            raise ValueError("preserve_row is required when parents are specified.")

        if preserve_row:
            assert_ptag(*parents)
            self._ptag = parents[0]._ptag
        else:
            self._ptag = new_ptag()

        super().__init__(value=value, distance=distance, parents=parents, root_name=root_name, children_type=children_type)

    def renew_ptag(self) -> None:
        self._ptag = new_ptag()

def assert_ptag(*prisoners: PrivPandasBase[Any]) -> None:
    if len(prisoners) > 0 and not all(prisoners[0]._ptag == p._ptag for p in prisoners):
        raise DPError("Row tags do not match")

class PrivPandasExclusiveDummy(PrivPandasBase[None]):
    def __init__(self, parents: Sequence[PrivPandasBase[Any]]):
        assert len(parents) > 0
        super().__init__(value=None, distance=parents[0].distance, parents=parents,
                         root_name=None, preserve_row=False, children_type="exclusive")

@egrpc.function
def total_max_distance(prisoners: list[SensitiveInt | SensitiveFloat]) -> realnum:
    return sum([x.distance for x in prisoners], start=RealExpr(0)).max()

@egrpc.dataclass
class Index:
    values : list[ElementType]
    name   : ElementType | None

    def to_pandas(self) -> _pd.Index[ElementType]:
        return _pd.Index(self.values, name=self.name)

@egrpc.dataclass
class MultiIndex:
    values : list[tuple[ElementType, ...]]
    names  : list[ElementType]

    def to_pandas(self) -> _pd.MultiIndex[ElementType]:
        return _pd.MultiIndex.from_tuples(self.values, names=self.names)

def pack_pandas_index(index: _pd.Index) -> Index | MultiIndex:
    if isinstance(index, _pd.MultiIndex):
        return MultiIndex(index.tolist(), index.names)
    else:
        return Index(index.tolist(), index.name)
