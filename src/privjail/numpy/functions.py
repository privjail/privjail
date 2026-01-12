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
