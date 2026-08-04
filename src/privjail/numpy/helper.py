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

import egrpc
import numpy as _np

from ..util import realnum
from .array import PrivNDArray
from .domain import NDArrayDomain


@egrpc.function
def clip_norm(arr: PrivNDArray, bound: realnum, ord: int | None = None) -> PrivNDArray:
    if bound <= 0:
        raise ValueError("`bound` must be positive.")

    ord_value = 2 if ord is None else ord
    if ord_value not in (1, 2):
        raise ValueError("`ord` must be 1, 2, or None.")

    # FIXME: support for privacy_axis > 0
    assert arr._privacy_axis == 0

    value_array = _np.asarray(arr._value, dtype=float)
    value_array = _np.where(_np.isfinite(value_array), value_array, 0.0)

    if value_array.size == 0:
        clipped = value_array
    elif value_array.ndim == 1:
        clipped = _np.clip(value_array, -float(bound), float(bound))
    else:
        nrows = value_array.shape[0]
        flat_rows = value_array.reshape(nrows, -1)
        norms = _np.linalg.norm(flat_rows, ord=ord_value, axis=1, keepdims=True)

        scales = _np.ones_like(norms, dtype=float)
        _np.divide(bound, norms, out=scales, where=norms > bound)
        scales = _np.where(_np.isfinite(norms), scales, 0.0)

        broadcast_shape = (nrows,) + (1,) * (value_array.ndim - 1)
        clipped = value_array * scales.reshape(broadcast_shape)

    norm_type = "l1" if ord_value == 1 else "l2"
    new_domain = NDArrayDomain(norm_type=norm_type, norm_bound=float(bound))
    return PrivNDArray(value          = clipped,
                       distance       = arr._distance,
                       privacy_axis   = arr._privacy_axis,
                       domain         = new_domain,
                       parents        = [arr],
                       keep_alignment = True)


@egrpc.function
def normalize(arr: PrivNDArray, ord: int | None = None) -> PrivNDArray:
    ord_value = 2 if ord is None else ord
    if ord_value not in (1, 2):
        raise ValueError("`ord` must be 1, 2, or None.")

    # FIXME: support for privacy_axis > 0
    assert arr._privacy_axis == 0

    eps = 1e-12
    value_array = _np.asarray(arr._value, dtype=float)
    value_array = _np.where(_np.isfinite(value_array), value_array, 0.0)

    if value_array.size == 0:
        normalized = value_array
    else:
        nrows = value_array.shape[0]
        flat_rows = value_array.reshape(nrows, -1)
        norms = _np.linalg.norm(flat_rows, ord=ord_value, axis=1, keepdims=True)
        broadcast_shape = (nrows,) + (1,) * (value_array.ndim - 1)
        normalized = value_array / (norms.reshape(broadcast_shape) + eps)
        normalized = _np.where(_np.isfinite(normalized), normalized, 0.0)

    norm_type = "l1" if ord_value == 1 else "l2"
    new_domain = NDArrayDomain(norm_type=norm_type, norm_bound=1.0, value_range=(-1.0, 1.0))
    return PrivNDArray(value          = normalized,
                       distance       = arr._distance,
                       privacy_axis   = arr._privacy_axis,
                       domain         = new_domain,
                       parents        = [arr],
                       keep_alignment = True)
