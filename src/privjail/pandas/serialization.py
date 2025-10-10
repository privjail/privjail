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
from typing import Any

import pandas as _pd

from .. import egrpc
from ..util import ElementType
from ..numpy.serialization import NDArrayPayload, pack_ndarray, unpack_ndarray
from .util import Index, MultiIndex, pack_pandas_index

@egrpc.dataclass
class SeriesPayload:
    values : NDArrayPayload
    index  : Index | MultiIndex
    name   : ElementType | None

@egrpc.dataclass
class DataFramePayload:
    values  : NDArrayPayload
    index   : Index | MultiIndex
    columns : Index | MultiIndex

def pack_series(series: _pd.Series[Any]) -> SeriesPayload:
    return SeriesPayload(values = pack_ndarray(series.to_numpy()),
                         index  = pack_pandas_index(series.index),
                         name   = series.name) # type: ignore[arg-type]

def unpack_series(payload: SeriesPayload) -> _pd.Series[Any]:
    return _pd.Series(data  = unpack_ndarray(payload.values),
                      index = payload.index.to_pandas(),
                      name  = payload.name)

def pack_dataframe(df: _pd.DataFrame) -> DataFramePayload:
    return DataFramePayload(values  = pack_ndarray(df.to_numpy()),
                            index   = pack_pandas_index(df.index),
                            columns = pack_pandas_index(df.columns))

def unpack_dataframe(payload: DataFramePayload) -> _pd.DataFrame:
    return _pd.DataFrame(data    = unpack_ndarray(payload.values),
                         index   = payload.index.to_pandas(),
                         columns = payload.columns.to_pandas())

egrpc.register_type(_pd.Series, SeriesPayload, pack_series, unpack_series)
egrpc.register_type(_pd.DataFrame, DataFramePayload, pack_dataframe, unpack_dataframe)
