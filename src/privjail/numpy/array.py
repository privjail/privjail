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
from ..util import DPError, floating, realnum
from ..array_base import PrivArrayBase, SensitiveDimInt
from ..realexpr import RealExpr
from ..accountants import Accountant
from ..prisoner import Prisoner, SensitiveFloat
from .domain import NDArrayDomain, ValueRange
from .util import PrivShape, infer_missing_dim, check_broadcast_distance_axis

def _negate_value_range(vr: ValueRange) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    new_lo = -hi if hi is not None else None
    new_hi = -lo if lo is not None else None
    return (new_lo, new_hi)

def _shift_value_range(vr: ValueRange, c: float) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    new_lo = lo + c if lo is not None else None
    new_hi = hi + c if hi is not None else None
    return (new_lo, new_hi)

def _rshift_value_range(c: float, vr: ValueRange) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    new_lo = c - hi if hi is not None else None
    new_hi = c - lo if lo is not None else None
    return (new_lo, new_hi)

def _rdiv_value_range(c: float, vr: ValueRange) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    if lo is None or hi is None:
        return None
    if lo <= 0 <= hi:
        return None
    if c == 0:
        return (0.0, 0.0)
    val1 = c / lo
    val2 = c / hi
    return (min(val1, val2), max(val1, val2))

def _scale_value_range(vr: ValueRange, c: float) -> ValueRange:
    if vr is None:
        return None
    lo, hi = vr
    if c >= 0:
        new_lo = lo * c if lo is not None else None
        new_hi = hi * c if hi is not None else None
    else:
        new_lo = hi * c if hi is not None else None
        new_hi = lo * c if lo is not None else None
    return (new_lo, new_hi)

def _add_value_ranges(vr1: ValueRange, vr2: ValueRange) -> ValueRange:
    if vr1 is None or vr2 is None:
        return None
    lo1, hi1 = vr1
    lo2, hi2 = vr2
    new_lo = lo1 + lo2 if lo1 is not None and lo2 is not None else None
    new_hi = hi1 + hi2 if hi1 is not None and hi2 is not None else None
    return (new_lo, new_hi)

def _sub_value_ranges(vr1: ValueRange, vr2: ValueRange) -> ValueRange:
    if vr1 is None or vr2 is None:
        return None
    lo1, hi1 = vr1
    lo2, hi2 = vr2
    new_lo = lo1 - hi2 if lo1 is not None and hi2 is not None else None
    new_hi = hi1 - lo2 if hi1 is not None and lo2 is not None else None
    return (new_lo, new_hi)

def _mul_value_ranges(vr1: ValueRange, vr2: ValueRange) -> ValueRange:
    if vr1 is None or vr2 is None:
        return None
    lo1, hi1 = vr1
    lo2, hi2 = vr2
    if lo1 is None or hi1 is None or lo2 is None or hi2 is None:
        return None
    products = [lo1 * lo2, lo1 * hi2, hi1 * lo2, hi1 * hi2]
    return (min(products), max(products))

def _div_value_ranges(vr1: ValueRange, vr2: ValueRange) -> ValueRange:
    if vr1 is None or vr2 is None:
        return None
    lo1, hi1 = vr1
    lo2, hi2 = vr2
    if lo1 is None or hi1 is None or lo2 is None or hi2 is None:
        return None
    if lo2 <= 0 <= hi2:
        return None
    quotients = [lo1 / lo2, lo1 / hi2, hi1 / lo2, hi1 / hi2]
    return (min(quotients), max(quotients))


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

        array.setflags(write=False) # make it immutable to avoid unexpected write to the same view

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
    def shape(self) -> PrivShape:
        return tuple(
            SensitiveDimInt(value          = int(dim),
                            distance       = self.distance,
                            axis_signature = self._axis_signature,
                            scale          = 1,
                            parents        = [self])
            if i == self.distance_axis else int(dim)
            for i, dim in enumerate(self._value.shape)
        )

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

    def reshape(self,
                *shape : int | SensitiveDimInt | PrivShape,
                order  : str = "C",
                ) -> PrivNDArray:
        if order != "C":
            raise ValueError("Only order='C' is supported for reshape.")

        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape_tuple: PrivShape = shape[0]
        else:
            shape_tuple = tuple(d for d in shape if not isinstance(d, tuple))

        return self._reshape_impl(shape_tuple)

    @egrpc.method
    def _reshape_impl(self, shape: PrivShape) -> PrivNDArray:
        output_shape = infer_missing_dim(self.shape, shape)

        priv_dims = [(i, d) for i, d in enumerate(output_shape) if isinstance(d, SensitiveDimInt)]
        if len(priv_dims) != 1:
            raise ValueError("Output shape must contain exactly one SensitiveDimInt.")
        output_axis, output_priv_dim = priv_dims[0]

        scale = output_priv_dim.scale
        for d in output_shape:
            val = d.scale if isinstance(d, SensitiveDimInt) else d
            if val <= 0:
                raise ValueError("All dimensions and scale must be positive for reshape.")

        if output_priv_dim.axis_signature != self.axis_signature:
            raise DPError("SensitiveDimInt's axis_signature does not match the array's axis_signature.")

        input_axis = self.distance_axis

        P_in = 1
        for d in self.shape[:input_axis]:
            assert isinstance(d, int)
            P_in *= d

        Q_in = 1
        for d in self.shape[input_axis + 1:]:
            assert isinstance(d, int)
            Q_in *= d

        P_out = 1
        for d in output_shape[:output_axis]:
            assert isinstance(d, int)
            P_out *= d

        Q_out = 1
        for d in output_shape[output_axis + 1:]:
            assert isinstance(d, int)
            Q_out *= d

        if not ((P_in == P_out and Q_in == Q_out * scale) or
                (P_in == P_out * scale and Q_in == Q_out)):
            raise DPError("Reshape would mix data across individuals.")

        final_shape = tuple(
            int(d._value) if isinstance(d, SensitiveDimInt) else d
            for d in output_shape
        )
        reshaped_arr = self._value.reshape(final_shape, order="C")

        return PrivNDArray(value                  = reshaped_arr,
                           distance               = self.distance * scale,
                           distance_axis          = output_axis,
                           domain                 = self._domain,
                           parents                = [self],
                           inherit_axis_signature = (scale == 1))

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
    def max(self, axis: int | None = None, keepdims: bool = False) -> PrivNDArray:
        if axis is None:
            raise DPError("max() with axis=None would collapse the distance axis.")

        if axis < 0:
            axis = self.ndim + axis

        if not (0 <= axis < self.ndim):
            raise ValueError(f"axis {axis} is out of bounds for array of dimension {self.ndim}.")

        if axis == self.distance_axis:
            raise DPError("max() along the distance axis is not allowed.")

        if keepdims:
            new_distance_axis = self.distance_axis
        else:
            if axis < self.distance_axis:
                new_distance_axis = self.distance_axis - 1
            else:
                new_distance_axis = self.distance_axis

        result = self._value.max(axis=axis, keepdims=keepdims)

        return PrivNDArray(value                  = result,
                           distance               = self.distance,
                           distance_axis          = new_distance_axis,
                           domain                 = NDArrayDomain(value_range=self._domain.value_range),
                           parents                = [self],
                           inherit_axis_signature = True)

    @egrpc.method
    def min(self, axis: int | None = None, keepdims: bool = False) -> PrivNDArray:
        if axis is None:
            raise DPError("min() with axis=None would collapse the distance axis.")

        if axis < 0:
            axis = self.ndim + axis

        if not (0 <= axis < self.ndim):
            raise ValueError(f"axis {axis} is out of bounds for array of dimension {self.ndim}.")

        if axis == self.distance_axis:
            raise DPError("min() along the distance axis is not allowed.")

        if keepdims:
            new_distance_axis = self.distance_axis
        else:
            if axis < self.distance_axis:
                new_distance_axis = self.distance_axis - 1
            else:
                new_distance_axis = self.distance_axis

        result = self._value.min(axis=axis, keepdims=keepdims)

        return PrivNDArray(value                  = result,
                           distance               = self.distance,
                           distance_axis          = new_distance_axis,
                           domain                 = NDArrayDomain(value_range=self._domain.value_range),
                           parents                = [self],
                           inherit_axis_signature = True)

    @egrpc.multimethod
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

    @__matmul__.register
    def _(self, other: PrivNDArray) -> SensitiveNDArray:
        if self.ndim != 2 or other.ndim != 2:
            raise NotImplementedError("matmul currently supports 2D operands only.")

        if self.distance_axis != 1:
            raise DPError("Left operand distance_axis must be 1 (the contracting axis).")
        if other.distance_axis != 0:
            raise DPError("Right operand distance_axis must be 0 (the contracting axis).")

        if self.axis_signature != other.axis_signature:
            raise DPError("axis_signature mismatch in PrivNDArray matmul.")

        if self.domain.norm_bound is None:
            raise DPError("Left operand norm_bound is not set.")
        if other.domain.norm_bound is None:
            raise DPError("Right operand norm_bound is not set.")

        new_distance = self.distance * self.domain.norm_bound * other.domain.norm_bound
        return SensitiveNDArray(value     = self._value @ other._value,
                                distance  = new_distance,
                                norm_type = "l2",
                                parents   = [self, other])

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

    @egrpc.method
    def __neg__(self) -> PrivNDArray:
        new_vr = _negate_value_range(self._domain.value_range)
        return PrivNDArray(value                  = -self._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
                           inherit_axis_signature = True)

    def _check_axis_aligned(self, other: PrivNDArray) -> None:
        if self.axis_signature != other.axis_signature:
            raise DPError("axis_signature mismatch in PrivNDArray operation.")
        check_broadcast_distance_axis(self.shape, other.shape)

    @egrpc.multimethod
    def __add__(self, other: realnum) -> PrivNDArray:
        new_vr = _shift_value_range(self._domain.value_range, float(other))
        return PrivNDArray(value                  = self._value + other,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
                           inherit_axis_signature = True)

    @__add__.register
    def _(self, other: PrivNDArray) -> PrivNDArray:
        self._check_axis_aligned(other)
        new_vr = _add_value_ranges(self._domain.value_range, other._domain.value_range)
        return PrivNDArray(value                  = self._value + other._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self, other],
                           inherit_axis_signature = True)

    @egrpc.method
    def __radd__(self, other: realnum) -> PrivNDArray:  # type: ignore[misc]
        new_vr = _shift_value_range(self._domain.value_range, float(other))
        return PrivNDArray(value                  = other + self._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
                           inherit_axis_signature = True)

    @egrpc.multimethod
    def __sub__(self, other: realnum) -> PrivNDArray:
        new_vr = _shift_value_range(self._domain.value_range, -float(other))
        return PrivNDArray(value                  = self._value - other,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
                           inherit_axis_signature = True)

    @__sub__.register
    def _(self, other: PrivNDArray) -> PrivNDArray:
        self._check_axis_aligned(other)
        new_vr = _sub_value_ranges(self._domain.value_range, other._domain.value_range)
        return PrivNDArray(value                  = self._value - other._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self, other],
                           inherit_axis_signature = True)

    @egrpc.method
    def __rsub__(self, other: realnum) -> PrivNDArray:  # type: ignore[misc]
        new_vr = _rshift_value_range(float(other), self._domain.value_range)
        return PrivNDArray(value                  = other - self._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
                           inherit_axis_signature = True)

    @egrpc.multimethod
    def __mul__(self, other: realnum) -> PrivNDArray:
        new_vr = _scale_value_range(self._domain.value_range, float(other))
        return PrivNDArray(value                  = self._value * other,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
                           inherit_axis_signature = True)

    @__mul__.register
    def _(self, other: PrivNDArray) -> PrivNDArray:
        self._check_axis_aligned(other)
        new_vr = _mul_value_ranges(self._domain.value_range, other._domain.value_range)
        return PrivNDArray(value                  = self._value * other._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self, other],
                           inherit_axis_signature = True)

    @egrpc.method
    def __rmul__(self, other: realnum) -> PrivNDArray:  # type: ignore[misc]
        new_vr = _scale_value_range(self._domain.value_range, float(other))
        return PrivNDArray(value                  = other * self._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
                           inherit_axis_signature = True)

    @egrpc.multimethod
    def __truediv__(self, other: realnum) -> PrivNDArray:
        if other == 0:
            raise ZeroDivisionError("division by zero")
        new_vr = _scale_value_range(self._domain.value_range, 1.0 / float(other))
        return PrivNDArray(value                  = self._value / other,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
                           inherit_axis_signature = True)

    @__truediv__.register
    def _(self, other: PrivNDArray) -> PrivNDArray:
        self._check_axis_aligned(other)
        new_vr = _div_value_ranges(self._domain.value_range, other._domain.value_range)
        return PrivNDArray(value                  = self._value / other._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self, other],
                           inherit_axis_signature = True)

    @egrpc.method
    def __rtruediv__(self, other: realnum) -> PrivNDArray:  # type: ignore[misc]
        new_vr = _rdiv_value_range(float(other), self._domain.value_range)
        return PrivNDArray(value                  = other / self._value,
                           distance               = self.distance,
                           distance_axis          = self.distance_axis,
                           domain                 = NDArrayDomain(value_range=new_vr),
                           parents                = [self],
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
