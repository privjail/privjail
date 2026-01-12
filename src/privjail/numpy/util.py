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
from types import EllipsisType
from typing import Any

import numpy as _np
import numpy.typing as _npt

from .. import egrpc
from ..array_base import SensitiveDimInt
from ..util import DPError

@egrpc.dataclass
class NDArrayPayload:
    data  : bytes
    shape : tuple[int, ...]
    dtype : str

    @classmethod
    def pack(cls, array: _npt.NDArray[Any]) -> NDArrayPayload:
        if not array.flags.c_contiguous:
            array = _np.ascontiguousarray(array)
        return cls(data  = array.tobytes(),
                   shape = array.shape,
                   dtype = str(array.dtype))

    def unpack(self) -> _npt.NDArray[Any]:
        dtype = _np.dtype(self.dtype)
        arr = _np.frombuffer(self.data, dtype=dtype)
        return arr.reshape(self.shape)

egrpc.register_type(_np.ndarray, NDArrayPayload)

PrivShape = tuple[int | SensitiveDimInt, ...]

IndexItem = int | slice | None | EllipsisType
NDIndex = IndexItem | tuple[IndexItem, ...]

def infer_missing_dim(input_shape: PrivShape, output_shape: PrivShape) -> PrivShape:
    if -1 not in output_shape:
        return output_shape

    input_priv_dim: SensitiveDimInt | None = None
    for dim in input_shape:
        if isinstance(dim, SensitiveDimInt):
            input_priv_dim = dim
            break
    assert input_priv_dim is not None

    output_priv_dim: SensitiveDimInt | None = None
    for dim in output_shape:
        if isinstance(dim, SensitiveDimInt):
            output_priv_dim = dim
            break

    input_PQ = 1
    for dim in input_shape:
        if isinstance(dim, int):
            input_PQ *= dim

    output_known = 1
    for dim in output_shape:
        if isinstance(dim, int) and dim != -1:
            output_known *= dim

    resolved: int | SensitiveDimInt
    if output_priv_dim is not None:
        scale = output_priv_dim.scale
        if scale * output_known <= 0 or input_PQ % (scale * output_known) != 0:
            raise ValueError("Cannot infer dimension for -1.")
        resolved = input_PQ // (scale * output_known)
    else:
        if input_PQ % output_known != 0:
            raise ValueError("Cannot infer dimension for -1.")
        scale = input_PQ // output_known
        resolved = input_priv_dim * scale

    return tuple(resolved if dim == -1 else dim for dim in output_shape)

def check_broadcast_distance_axis(shape1: PrivShape, shape2: PrivShape) -> None:
    max_ndim = max(len(shape1), len(shape2))
    padded1: PrivShape = (1,) * (max_ndim - len(shape1)) + shape1
    padded2: PrivShape = (1,) * (max_ndim - len(shape2)) + shape2

    for d1, d2 in zip(padded1, padded2):
        if isinstance(d1, SensitiveDimInt) and isinstance(d2, SensitiveDimInt):
            continue  # Same axis_signature means same size
        if isinstance(d1, int) and isinstance(d2, int):
            if d1 == d2 or d1 == 1 or d2 == 1:
                continue
            raise DPError(f"Shapes are not broadcastable: {d1} vs {d2}.")
        # SensitiveDimInt vs int: always NG
        raise DPError("Broadcasting to/from distance_axis is not allowed.")
