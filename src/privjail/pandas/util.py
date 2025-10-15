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
from ..numpy.util import NDArrayPayload

ColumnType = str
ColumnsType = list[ColumnType] | tuple[ColumnType, ...]

@egrpc.dataclass
class IndexPayload:
    values : Sequence[ElementType]
    name   : ElementType | None

    @classmethod
    def pack(cls, value: _pd.Index[Any]) -> IndexPayload:
        return cls(value.tolist(), value.name)

    def unpack(self) -> _pd.Index[Any]:
        return _pd.Index(self.values, name=self.name)

egrpc.register_type(_pd.Index, IndexPayload)

@egrpc.dataclass
class MultiIndexPayload:
    values : Sequence[tuple[ElementType, ...]]
    names  : Sequence[ElementType | None]

    @classmethod
    def pack(cls, value: _pd.MultiIndex) -> MultiIndexPayload:
        return cls(value.tolist(), list(value.names))

    def unpack(self) -> _pd.MultiIndex:
        return _pd.MultiIndex.from_tuples(self.values, names=list(self.names))

egrpc.register_type(_pd.MultiIndex, MultiIndexPayload)

def pack_pandas_index(index: _pd.Index[Any] | _pd.MultiIndex) -> IndexPayload | MultiIndexPayload:
    if isinstance(index, _pd.MultiIndex):
        return MultiIndexPayload.pack(index)
    elif isinstance(index, _pd.Index):
        return IndexPayload.pack(index)
    else:
        raise TypeError

@egrpc.dataclass
class SeriesPayload:
    values : NDArrayPayload
    index  : IndexPayload | MultiIndexPayload
    name   : ElementType | None

    @classmethod
    def pack(cls, value: _pd.Series[Any]) -> SeriesPayload:
        return cls(values = NDArrayPayload.pack(value.to_numpy()),
                   index  = pack_pandas_index(value.index),
                   name   = value.name) # type: ignore

    def unpack(self) -> _pd.Series[Any]:
        return _pd.Series(data  = self.values.unpack(),
                          index = self.index.unpack(),
                          name  = self.name)

egrpc.register_type(_pd.Series, SeriesPayload)

@egrpc.dataclass
class DataFramePayload:
    values  : NDArrayPayload
    index   : IndexPayload | MultiIndexPayload
    columns : IndexPayload | MultiIndexPayload

    @classmethod
    def pack(cls, value: _pd.DataFrame) -> DataFramePayload:
        return DataFramePayload(values  = NDArrayPayload.pack(value.to_numpy()),
                                index   = pack_pandas_index(value.index),
                                columns = pack_pandas_index(value.columns))

    def unpack(self) -> _pd.DataFrame:
        return _pd.DataFrame(data    = self.values.unpack(),
                             index   = self.index.unpack(),
                             columns = self.columns.unpack())

egrpc.register_type(_pd.DataFrame, DataFramePayload)
