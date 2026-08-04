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

import secrets
from typing import Any, TypeGuard

import numpy as _np
import numpy.typing as _npt

class DPError(Exception):
    pass

integer = int | _np.integer[Any]
floating = float | _np.floating[Any]
realnum = integer | floating

def is_integer(x: Any) -> TypeGuard[integer]:
    return isinstance(x, (int, _np.integer))

def is_floating(x: Any) -> TypeGuard[floating]:
    return isinstance(x, (float, _np.floating))

def is_realnum(x: Any) -> TypeGuard[realnum]:
    return is_integer(x) or is_floating(x)

def _secure_poisson_mask(size: int, q: float) -> _npt.NDArray[_np.bool_]:
    if q == 1.0:
        return _np.ones(size, dtype=_np.bool_)
    threshold = min(int(q * (1 << 64)), (1 << 64) - 1)
    samples = _np.frombuffer(
        secrets.token_bytes(8 * size),
        dtype=_np.uint64,
    )
    return samples < _np.uint64(threshold)

ElementType = realnum | str | bool
