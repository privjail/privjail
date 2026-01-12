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
from typing import Any, Sequence, TypeGuard

import numpy as _np
import numpy.typing as _npt

from .. import egrpc
from ..util import DPError, realnum
from .array import PrivNDArray, SensitiveNDArray
from .domain import NDArrayDomain, ValueRange

def _is_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    if isinstance(value, _np.ndarray):
        return value.ndim > 0
    return isinstance(value, Sequence)

@egrpc.function
def histogram(a     : PrivNDArray,
              bins  : int | Sequence[realnum]        = 10,
              range : tuple[realnum, realnum] | None = None,
              ) -> tuple[SensitiveNDArray, _npt.NDArray[_np.floating[Any]]]:
    if a.ndim != 1:
        if not all(dim == 1 for dim in a._value.shape[1:]):
            raise ValueError("`a` must be 1-D or have trailing singleton dimensions.")

    if _is_sequence(bins):
        if range is not None:
            raise ValueError("`range` must be None when explicit bin edges are provided.")
    else:
        if range is None:
            raise DPError("`range` must be specified when bins are given as counts.")

    hist, edges = _np.histogram(a._value, bins=bins, range=range) # type: ignore

    sensitive_hist = SensitiveNDArray(value     = hist,
                                      distance  = a.distance,
                                      norm_type = "l1",
                                      parents   = [a])

    return sensitive_hist, edges

@egrpc.function
def histogramdd(sample : PrivNDArray,
                bins   : int | Sequence[int] | Sequence[Sequence[realnum]] = 10,
                range  : Sequence[tuple[realnum, realnum]] | None          = None,
                ) -> tuple[SensitiveNDArray, tuple[_npt.NDArray[_np.floating[Any]], ...]]:
    if sample.ndim == 1:
        dim = 1
    elif sample.ndim == 2:
        dim = sample._value.shape[1]
    else:
        raise ValueError("`sample` must be (N, D) array")

    if _is_sequence(bins) and len(bins) > 0 and _is_sequence(bins[0]):
        if range is not None:
            raise ValueError("`range` must be None when explicit bin edges are provided.")
        if len(bins) != dim:
            raise ValueError("`bins` length must match the dimensionality of the sample.")
    else:
        if range is None:
            raise DPError("`range` must be specified when bins are given as counts.")
        if len(range) != dim:
            raise ValueError("`range` length must match the dimensionality of the sample.")
        if _is_sequence(bins) and len(bins) != dim:
            raise ValueError("`bins` length must match the dimensionality of the sample.")

    hist, edges = _np.histogramdd(sample._value, bins=bins, range=range) # type: ignore

    sensitive_hist = SensitiveNDArray(value     = hist,
                                      distance  = sample.distance,
                                      norm_type = "l1",
                                      parents   = [sample])

    return sensitive_hist, edges

def _maximum_value_ranges(vr1: ValueRange, vr2: ValueRange) -> ValueRange:
    if vr1 is None or vr2 is None:
        return None
    lo1, hi1 = vr1
    lo2, hi2 = vr2
    new_lo = max(lo1, lo2) if lo1 is not None and lo2 is not None else None
    new_hi = max(hi1, hi2) if hi1 is not None and hi2 is not None else None
    return (new_lo, new_hi)

def _maximum_value_range_scalar(vr: ValueRange, c: float) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    new_lo = max(lo, c) if lo is not None else None
    new_hi = max(hi, c) if hi is not None else None
    return (new_lo, new_hi)

def _minimum_value_ranges(vr1: ValueRange, vr2: ValueRange) -> ValueRange:
    if vr1 is None or vr2 is None:
        return None
    lo1, hi1 = vr1
    lo2, hi2 = vr2
    new_lo = min(lo1, lo2) if lo1 is not None and lo2 is not None else None
    new_hi = min(hi1, hi2) if hi1 is not None and hi2 is not None else None
    return (new_lo, new_hi)

def _minimum_value_range_scalar(vr: ValueRange, c: float) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    new_lo = min(lo, c) if lo is not None else None
    new_hi = min(hi, c) if hi is not None else None
    return (new_lo, new_hi)

def _exp_value_range(vr: ValueRange) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    new_lo = _np.exp(lo) if lo is not None else None
    new_hi = _np.exp(hi) if hi is not None else None
    return (new_lo, new_hi)

def _log_value_range(vr: ValueRange) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    if lo is not None and lo <= 0:
        return None
    new_lo = _np.log(lo) if lo is not None else None
    new_hi = _np.log(hi) if hi is not None else None
    return (new_lo, new_hi)

@egrpc.function
def maximum(x1: PrivNDArray, x2: PrivNDArray | realnum) -> PrivNDArray:
    if isinstance(x2, PrivNDArray):
        x1._check_axis_aligned(x2)
        new_vr = _maximum_value_ranges(x1.domain.value_range, x2.domain.value_range)
        return PrivNDArray(value                  = _np.maximum(x1._value, x2._value),
                           distance               = x1.distance,
                           distance_axis          = x1.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [x1, x2],
                           inherit_axis_signature = True)
    else:
        new_vr = _maximum_value_range_scalar(x1.domain.value_range, float(x2))
        return PrivNDArray(value                  = _np.maximum(x1._value, x2),
                           distance               = x1.distance,
                           distance_axis          = x1.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [x1],
                           inherit_axis_signature = True)

@egrpc.function
def minimum(x1: PrivNDArray, x2: PrivNDArray | realnum) -> PrivNDArray:
    if isinstance(x2, PrivNDArray):
        x1._check_axis_aligned(x2)
        new_vr = _minimum_value_ranges(x1.domain.value_range, x2.domain.value_range)
        return PrivNDArray(value                  = _np.minimum(x1._value, x2._value),
                           distance               = x1.distance,
                           distance_axis          = x1.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [x1, x2],
                           inherit_axis_signature = True)
    else:
        new_vr = _minimum_value_range_scalar(x1.domain.value_range, float(x2))
        return PrivNDArray(value                  = _np.minimum(x1._value, x2),
                           distance               = x1.distance,
                           distance_axis          = x1.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [x1],
                           inherit_axis_signature = True)

@egrpc.function
def exp(x: PrivNDArray) -> PrivNDArray:
    new_vr = _exp_value_range(x.domain.value_range)
    return PrivNDArray(value                  = _np.exp(x._value),
                       distance               = x.distance,
                       distance_axis          = x.distance_axis,
                       domain                 = NDArrayDomain(value_range=new_vr),
                       parents                = [x],
                       inherit_axis_signature = True)

@egrpc.function
def log(x: PrivNDArray) -> PrivNDArray:
    new_vr = _log_value_range(x.domain.value_range)
    return PrivNDArray(value                  = _np.log(x._value),
                       distance               = x.distance,
                       distance_axis          = x.distance_axis,
                       domain                 = NDArrayDomain(value_range=new_vr),
                       parents                = [x],
                       inherit_axis_signature = True)
