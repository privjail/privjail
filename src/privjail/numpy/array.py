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
from ..util import DPError, floating, realnum
from ..array_base import PrivArrayBase
from ..realexpr import RealExpr
from ..accountants import Accountant
from ..prisoner import Prisoner, SensitiveFloat, SensitiveInt
from .domain import NDArrayDomain

@egrpc.remoteclass
class PrivNDArray(PrivArrayBase[_npt.NDArray[_np.floating[Any]]]):
    __array_priority__ = 1000  # To prevent NumPy from treating PrivNDArray as a scalar operand
    _domain: NDArrayDomain

    def __init__(self,
                 value                  : Any,
                 distance               : RealExpr,
                 distance_axis          : int,
                 domain                 : NDArrayDomain | None         = None,
                 *,
                 parents                : Sequence[PrivArrayBase[Any]] = [],
                 accountant             : Accountant[Any] | None       = None,
                 inherit_axis_signature : bool                         = False,
                 ) -> None:
        array = _np.asarray(value)
        if not _np.issubdtype(array.dtype, _np.number):
            raise TypeError("PrivNDArray requires a numeric dtype.")

        self._domain = domain if domain is not None else NDArrayDomain()

        if distance_axis < 0:
            distance_axis += array.ndim
        if not 0 <= distance_axis < array.ndim:
            raise ValueError("distance_axis is out of bounds for the array dimension.")

        super().__init__(value                  = array,
                         distance               = distance,
                         distance_axis          = distance_axis,
                         parents                = list(parents),
                         accountant             = accountant,
                         inherit_axis_signature = inherit_axis_signature)

    @egrpc.property
    def shape(self) -> tuple[SensitiveInt | int, ...]:
        return tuple(SensitiveInt(value=int(dim), distance=self.distance, parents=[self]) if i == self.distance_axis else int(dim)
                     for i, dim in enumerate(self._value.shape))

    @egrpc.property
    def ndim(self) -> int:
        return self._value.ndim

    @egrpc.property
    def domain(self) -> NDArrayDomain:
        return self._domain

    @egrpc.property
    def T(self) -> PrivNDArray:
        return self.transpose()

    @egrpc.method
    def transpose(self, axes: tuple[int, ...] | list[int] | None = None) -> PrivNDArray:
        if axes is None:
            axes_tuple = tuple(reversed(range(self.ndim)))
        else:
            axes_tuple = tuple(axes)
            if len(axes_tuple) != self.ndim:
                raise ValueError("axes tuple must include each axis exactly once.")

        if sorted(axes_tuple) != list(range(self.ndim)):
            raise ValueError("axes must be a permutation of current axes.")

        return PrivNDArray(value                  = self._value.transpose(axes_tuple),
                           distance               = self.distance,
                           distance_axis          = axes_tuple.index(self.distance_axis),
                           domain                 = self._domain,
                           parents                = [self],
                           inherit_axis_signature = True)

    @egrpc.method
    def swapaxes(self, axis1: int, axis2: int) -> PrivNDArray:
        axis1 = axis1 + self.ndim if axis1 < 0 else axis1
        axis2 = axis2 + self.ndim if axis2 < 0 else axis2
        if not (0 <= axis1 < self.ndim) or not (0 <= axis2 < self.ndim):
            raise ValueError("axis index out of bounds.")

        if axis1 == self.distance_axis:
            new_distance_axis = axis2
        elif axis2 == self.distance_axis:
            new_distance_axis = axis1
        else:
            new_distance_axis = self.distance_axis

        return PrivNDArray(value                  = self._value.swapaxes(axis1, axis2),
                           distance               = self.distance,
                           distance_axis          = new_distance_axis,
                           domain                 = self._domain,
                           parents                = [self],
                           inherit_axis_signature = True)

    @egrpc.method
    def clip_norm(self, bound: realnum, ord: int | None = None) -> PrivNDArray:
        if bound <= 0:
            raise ValueError("`bound` must be positive.")

        ord_value = 2 if ord is None else ord
        if ord_value not in (1, 2):
            raise ValueError("`ord` must be 1, 2, or None.")

        # FIXME: support for distance_axis > 0
        assert self.distance_axis == 0

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
        return PrivNDArray(value                  = clipped,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = new_domain,
                           parents                = [self],
                           inherit_axis_signature = True)

    @egrpc.method
    def sum(self, axis: int | None = None) -> SensitiveFloat | SensitiveNDArray:
        if not ((axis is None and self.ndim == 1) or axis == self.distance_axis):
            raise NotImplementedError("sum() currently supports summing along the distance axis only.")

        if self._domain.norm_bound is None:
            raise DPError("Norm bound is not set. Use clip_norm() before summing along the distance axis.")

        new_distance = self.distance * float(self._domain.norm_bound)
        result = self._value.sum(axis=axis)

        if self.ndim == 1:
            return SensitiveFloat(float(result), distance=new_distance, parents=[self])
        else:
            return SensitiveNDArray(value     = result,
                                    distance  = new_distance,
                                    norm_type = self._domain.norm_type,
                                    parents   = [self])

    @egrpc.method
    def __matmul__(self, other: _npt.NDArray[Any]) -> PrivNDArray:
        if self.ndim != 2 or other.ndim != 2:
            raise NotImplementedError("matmul currently supports 2D operands only.")

        if self.distance_axis != 0:
            raise DPError("Matmul would contract the distance axis.")

        if self.shape[1] != other.shape[0]:
            raise ValueError("Shape mismatch for matrix multiplication.")

        return PrivNDArray(value                  = self._value @ other,
                           distance               = self.distance,
                           domain                 = NDArrayDomain(),
                           parents                = [self],
                           distance_axis          = self.distance_axis,
                           inherit_axis_signature = True)

    @egrpc.method
    def __rmatmul__(self, other: _npt.NDArray[Any]) -> PrivNDArray: # type: ignore[misc]
        if self.ndim != 2 or other.ndim != 2:
            raise NotImplementedError("matmul currently supports 2D operands only.")

        if self.distance_axis != 1:
            raise DPError("Matmul would contract the distance axis.")

        if self.shape[0] != other.shape[1]:
            raise ValueError("Shape mismatch for matrix multiplication.")

        return PrivNDArray(value                  = other @ self._value,
                           distance               = self.distance,
                           domain                 = NDArrayDomain(),
                           parents                = [self],
                           distance_axis          = self.distance_axis,
                           inherit_axis_signature = True)

@egrpc.remoteclass
class SensitiveNDArray(Prisoner[_npt.NDArray[_np.floating[Any]]]):
    __array_priority__ = 1000  # To prevent NumPy from treating SensitiveNDArray as a scalar operand
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

    @egrpc.method
    def __neg__(self) -> SensitiveNDArray:
        return SensitiveNDArray(value     = -self._value,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    @egrpc.multimethod
    def __add__(self, other: realnum) -> SensitiveNDArray:
        return SensitiveNDArray(value     = self._value + other,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    # FIXME: type mismatch error with numpy>=2.2.0
    @__add__.register
    def _(self, other: _npt.NDArray[Any]) -> SensitiveNDArray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch in SensitiveNDArray addition.")
        return SensitiveNDArray(value     = self._value + other,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    @__add__.register
    def _(self, other: SensitiveNDArray) -> SensitiveNDArray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch in SensitiveNDArray addition.")
        if self.norm_type != other.norm_type:
            raise ValueError("Norm type mismatch in SensitiveNDArray addition.")
        return SensitiveNDArray(value     = self._value + other._value,
                                distance  = self.distance + other.distance,
                                norm_type = self.norm_type,
                                parents   = [self, other])

    @egrpc.multimethod
    def __radd__(self, other: realnum) -> SensitiveNDArray:  # type: ignore[misc]
        return SensitiveNDArray(value     = other + self._value,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    @__radd__.register
    def _(self, other: _npt.NDArray[Any]) -> SensitiveNDArray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch in SensitiveNDArray addition.")
        return SensitiveNDArray(value     = other + self._value,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    @egrpc.multimethod
    def __sub__(self, other: realnum) -> SensitiveNDArray:
        return SensitiveNDArray(value     = self._value - other,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    @__sub__.register
    def _(self, other: _npt.NDArray[Any]) -> SensitiveNDArray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch in SensitiveNDArray subtraction.")
        return SensitiveNDArray(value     = self._value - other,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    @__sub__.register
    def _(self, other: SensitiveNDArray) -> SensitiveNDArray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch in SensitiveNDArray subtraction.")
        if self.norm_type != other.norm_type:
            raise ValueError("Norm type mismatch in SensitiveNDArray subtraction.")
        return SensitiveNDArray(value     = self._value - other._value,
                                distance  = self.distance + other.distance,
                                norm_type = self.norm_type,
                                parents   = [self, other])

    @egrpc.multimethod
    def __rsub__(self, other: realnum) -> SensitiveNDArray:  # type: ignore[misc]
        return SensitiveNDArray(value     = other - self._value,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    @__rsub__.register
    def _(self, other: _npt.NDArray[Any]) -> SensitiveNDArray:
        if self.shape != other.shape:
            raise ValueError("Shape mismatch in SensitiveNDArray subtraction.")
        return SensitiveNDArray(value     = other - self._value,
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    # TODO: support Literal type in egrpc
    @egrpc.method
    def flatten(self, order: str = "C") -> SensitiveNDArray:
        return SensitiveNDArray(value     = self._value.flatten(order=order), # type: ignore
                                distance  = self.distance,
                                norm_type = self.norm_type,
                                parents   = [self])

    def reveal(self,
               *,
               eps   : floating | None = None,
               delta : floating | None = None,
               rho   : floating | None = None,
               scale : floating | None = None,
               mech  : str             = "laplace",
               ) -> _npt.NDArray[_np.floating[Any]]:
        if mech == "laplace":
            from ..mechanism import laplace_mechanism
            return laplace_mechanism(self,
                                     eps   = float(eps)   if eps   is not None else None,
                                     scale = float(scale) if scale is not None else None)
        elif mech == "gaussian":
            from ..mechanism import gaussian_mechanism
            return gaussian_mechanism(self,
                                      eps   = float(eps)   if eps   is not None else None,
                                      delta = float(delta) if delta is not None else None,
                                      rho   = float(rho)   if rho   is not None else None,
                                      scale = float(scale) if scale is not None else None)
        else:
            raise ValueError(f"Unknown DP mechanism: '{mech}'")

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
