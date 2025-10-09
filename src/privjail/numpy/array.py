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

import numpy as _np
import numpy.typing as _npt

from .. import egrpc
from ..util import realnum
from ..array_base import PrivArrayBase
from ..realexpr import RealExpr
from ..accountants import Accountant
from ..prisoner import Prisoner, SensitiveFloat, SensitiveInt
from ..util import DPError
from .domain import NDArrayDomain

@egrpc.remoteclass
class PrivNDArray(PrivArrayBase[_npt.NDArray[_np.floating[Any]]]):
    _domain: NDArrayDomain

    def __init__(self,
                 value        : Any,
                 distance     : RealExpr,
                 domain       : NDArrayDomain | None         = None,
                 *,
                 parents      : Sequence[PrivArrayBase[Any]] = [],
                 accountant   : Accountant[Any] | None       = None,
                 preserve_row : bool | None                  = None,
                 ) -> None:
        array = _np.asarray(value)
        if not _np.issubdtype(array.dtype, _np.number):
            raise TypeError("PrivNDArray requires a numeric dtype.")

        self._domain = domain if domain is not None else NDArrayDomain()

        super().__init__(value        = array,
                         distance     = distance,
                         parents      = list(parents),
                         accountant   = accountant,
                         preserve_row = preserve_row)

    @egrpc.property
    def shape(self) -> tuple[SensitiveInt | int, ...]:
        dims = self._value.shape
        assert len(dims) > 0
        nrows = SensitiveInt(value=dims[0], distance=self.distance, parents=[self])
        return (nrows, *dims[1:])

    @egrpc.property
    def ndim(self) -> int:
        return self._value.ndim

    @egrpc.property
    def domain(self) -> NDArrayDomain:
        return self._domain

    @egrpc.method
    def clip_norm(self, bound: realnum, ord: int | None = None) -> PrivNDArray:
        if bound <= 0:
            raise ValueError("`bound` must be positive.")

        ord_value = 2 if ord is None else ord
        if ord_value not in (1, 2):
            raise ValueError("`ord` must be 1, 2, or None.")

        value_array = _np.asarray(self._value, dtype=float)

        if value_array.ndim == 1:
            clipped = _np.clip(value_array, -float(bound), float(bound))
        else:
            nrows = value_array.shape[0]
            flat_rows = value_array.reshape(nrows, -1)
            norms = _np.linalg.norm(flat_rows, ord=ord_value, axis=1, keepdims=True)

            scales = _np.ones_like(norms, dtype=float)
            _np.divide(bound, norms, out=scales, where=norms > bound)

            broadcast_shape = (nrows,) + (1,) * (value_array.ndim - 1)
            clipped = value_array * scales.reshape(broadcast_shape)

        norm_label = "l1" if ord_value == 1 else "l2"
        new_domain = NDArrayDomain(norm_type=norm_label, norm_bound=float(bound))
        return PrivNDArray(value        = clipped,
                           distance     = self.distance,
                           domain       = new_domain,
                           parents      = [self],
                           preserve_row = True)

    @egrpc.method
    def sum(self, axis: int | None = None) -> SensitiveFloat | SensitiveNDArray:
        if axis != 0:
            raise NotImplementedError("sum() currently supports axis=0 only.")

        if self._domain.norm_bound is None:
            raise DPError("Norm bound is not set. Use clip_norm() before summing along axis 0.")

        new_distance = self.distance * float(self._domain.norm_bound)
        result = self._value.sum(axis=0)

        if self._value.ndim == 1:
            return SensitiveFloat(float(result), distance=new_distance, parents=[self])
        else:
            return SensitiveNDArray(value     = result,
                                    distance  = new_distance,
                                    norm_type = self._domain.norm_type,
                                    parents   = [self])

@egrpc.remoteclass
class SensitiveNDArray(Prisoner[_npt.NDArray[_np.floating[Any]]]):
    _norm_type: str

    def __init__(self,
                 value      : Any,
                 distance   : RealExpr,
                 norm_type  : str                     = "l1",
                 *,
                 parents    : Sequence[Prisoner[Any]] = [],
                 accountant : Accountant[Any] | None  = None,
                 ) -> None:
        array = _np.asarray(value, dtype=float)
        self._norm_type = norm_type

        super().__init__(value=array, distance=distance, parents=parents, accountant=accountant)

    @egrpc.property
    def shape(self) -> tuple[int, ...]:
        return self._value.shape # type: ignore

    @egrpc.property
    def ndim(self) -> int:
        return self._value.ndim

    @egrpc.property
    def norm_type(self) -> str:
        return self._norm_type
