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

import numpy as _np
import numpy.typing as _npt

from .. import egrpc

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
