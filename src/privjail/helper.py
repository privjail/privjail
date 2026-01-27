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
from typing import overload

import numpy as _np
import numpy.typing as _npt

from .util import realnum, DPError
from .numpy import PrivNDArray, NDArrayDomain
from .alignment import new_alignment_signature
from . import egrpc

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

    if value_array.size == 0:
        normalized = value_array
    elif value_array.ndim == 1:
        norm = _np.linalg.norm(value_array, ord=ord_value)
        normalized = value_array / (norm + eps)
    else:
        nrows = value_array.shape[0]
        flat_rows = value_array.reshape(nrows, -1)
        norms = _np.linalg.norm(flat_rows, ord=ord_value, axis=1, keepdims=True)
        broadcast_shape = (nrows,) + (1,) * (value_array.ndim - 1)
        normalized = value_array / (norms.reshape(broadcast_shape) + eps)

    norm_type = "l1" if ord_value == 1 else "l2"
    new_domain = NDArrayDomain(norm_type=norm_type, norm_bound=1.0, value_range=(-1.0, 1.0))
    return PrivNDArray(value          = normalized,
                       distance       = arr._distance,
                       privacy_axis   = arr._privacy_axis,
                       domain         = new_domain,
                       parents        = [arr],
                       keep_alignment = True)

@overload
def sample(array: PrivNDArray, /, *, q: float, method: str = "poisson") -> PrivNDArray: ...
@overload
def sample(*arrays: PrivNDArray, q: float, method: str = "poisson") -> tuple[PrivNDArray, ...]: ...

def sample(*arrays: PrivNDArray, q: float, method: str = "poisson") -> PrivNDArray | tuple[PrivNDArray, ...]:
    result = _sample_impl(arrays, q, method)
    if len(arrays) == 1:
        return result[0]
    return result

@egrpc.function
def _sample_impl(arrays: tuple[PrivNDArray, ...], q: float, method: str) -> tuple[PrivNDArray, ...]:
    if len(arrays) == 0:
        raise ValueError("At least one array is required.")

    if not (0.0 < q <= 1.0):
        raise ValueError("Sampling rate q must be in (0, 1].")

    if method != "poisson":
        raise ValueError(f"Unknown sampling method: '{method}'")

    first = arrays[0]

    if not all(arr.alignment_signature == first.alignment_signature for arr in arrays[1:]):
        raise DPError("All arrays must have the same alignment_signature.")

    # FIXME: support for privacy_axis > 0
    assert all(arr._privacy_axis == 0 for arr in arrays)

    effective_max_distance = float(first._distance.max())
    if effective_max_distance != 1.0:
        raise DPError("Subsampling requires adjacent databases (max_distance=1)")

    n = first._value.shape[0]
    mask: _npt.NDArray[_np.bool_] = _np.random.random(n) < q

    parent_accountant = first.accountant
    child_accountant = parent_accountant.create_subsampling_accountant(q)

    sig = new_alignment_signature()
    results: list[PrivNDArray] = []
    for arr in arrays:
        out = PrivNDArray(value          = arr._value[mask],
                          distance       = arr._distance,
                          privacy_axis   = arr._privacy_axis,
                          domain         = arr.domain,
                          parents        = [arr],
                          accountant     = child_accountant,
                          keep_alignment = False)
        out._alignment_signature = sig
        results.append(out)

    return tuple(results)
