# Copyright 2026 Shumpei Shiina.
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

from collections.abc import Sequence
from typing import Any, overload

import egrpc
import jax
import jax.numpy as jnp

from ..accountants import Accountant
from ..array_base import PrivArrayBase, SensitiveDimInt
from ..numpy import NDArrayDomain, PrivNDArray
from ..numpy.util import PrivShape
from ..prisoner import Prisoner
from ..realexpr import RealExpr
from .domain import ArrayDomain


@egrpc.remoteclass
class PrivArray(PrivArrayBase[jax.Array]):
    __array_priority__ = 1000
    _domain: ArrayDomain

    def __init__(
        self,
        value: Any,
        distance: RealExpr,
        privacy_axis: int,
        domain: ArrayDomain | None = None,
        *,
        parents: Sequence[PrivArrayBase[Any]] = (),
        accountant: Accountant[Any] | None = None,
        keep_alignment: bool = False,
    ) -> None:
        array = jnp.asarray(value)
        if not (
            jnp.issubdtype(array.dtype, jnp.number)
            or jnp.issubdtype(array.dtype, jnp.bool_)
        ):
            raise TypeError("PrivArray requires a numeric or boolean dtype.")

        if privacy_axis < 0:
            privacy_axis += array.ndim
        if not 0 <= privacy_axis < array.ndim:
            raise ValueError("privacy_axis is out of bounds for the array rank.")

        self._domain = ArrayDomain() if domain is None else domain
        super().__init__(
            value=array,
            distance=distance,
            privacy_axis=privacy_axis,
            parents=parents,
            accountant=accountant,
            keep_alignment=keep_alignment,
        )

    @egrpc.property
    def shape(self) -> PrivShape:
        return tuple(
            SensitiveDimInt(
                value=int(dim),
                distance=self._distance,
                alignment_signature=self._alignment_signature,
                parents=[self],
            )
            if axis == self._privacy_axis
            else int(dim)
            for axis, dim in enumerate(self._value.shape)
        )

    @egrpc.property
    def ndim(self) -> int:
        return int(self._value.ndim)

    @egrpc.property
    def dtype(self) -> str:
        return str(self._value.dtype)

    @egrpc.property
    def weak_type(self) -> bool:
        return bool(getattr(self._value, "weak_type", False))

    @egrpc.property
    def domain(self) -> NDArrayDomain:
        norm_bound = self._domain.norm_bound
        return NDArrayDomain(
            norm_type=self._domain.norm_type,
            norm_bound=(
                None
                if norm_bound is None
                else float(norm_bound.max())
            ),
            value_range=self._domain.value_range,
        )

    def __add__(self, other: Any) -> PrivArray:
        from .primitives import add

        result: PrivArray = add(
            self,
            other if isinstance(other, PrivArray) else jnp.asarray(other),
        )
        return result

    def __radd__(self, other: Any) -> PrivArray:
        from .primitives import add

        result: PrivArray = add(
            other if isinstance(other, PrivArray) else jnp.asarray(other),
            self,
        )
        return result

    def __mul__(self, other: Any) -> PrivArray:
        from .primitives import mul

        result: PrivArray = mul(
            self,
            other if isinstance(other, PrivArray) else jnp.asarray(other),
        )
        return result

    def __rmul__(self, other: Any) -> PrivArray:
        from .primitives import mul

        result: PrivArray = mul(
            other if isinstance(other, PrivArray) else jnp.asarray(other),
            self,
        )
        return result

    def __truediv__(self, other: Any) -> PrivArray:
        from .primitives import div

        result: PrivArray = div(
            self,
            other if isinstance(other, PrivArray) else jnp.asarray(other),
        )
        return result

    def __matmul__(
        self,
        other: Any,
    ) -> PrivArray | SensitiveArray:
        from .primitives import dot_general

        if self.ndim != 2:
            raise NotImplementedError(
                "PrivArray matmul currently supports 2D operands only."
            )
        rhs = other if isinstance(other, PrivArray) else jnp.asarray(other)
        if rhs.ndim != 2:
            raise NotImplementedError(
                "PrivArray matmul currently supports 2D operands only."
            )
        result: PrivArray | SensitiveArray = dot_general(
            self,
            rhs,
            (((1,), (0,)), ((), ())),
            None,
            None,
            None,
        )
        return result

    @overload
    def reshape(
        self,
        shape: tuple[int | SensitiveDimInt, ...],
        /,
    ) -> PrivArray: ...

    @overload
    def reshape(self, *shape: int | SensitiveDimInt) -> PrivArray: ...

    def reshape(self, *shape: Any) -> PrivArray:
        from .primitives import reshape

        if len(shape) == 1 and isinstance(shape[0], tuple):
            new_sizes = shape[0]
        else:
            new_sizes = tuple(shape)
        result: PrivArray = reshape(self, new_sizes, None)
        return result

    @overload
    def transpose(self, axes: tuple[int, ...], /) -> PrivArray: ...

    @overload
    def transpose(self, *axes: int) -> PrivArray: ...

    def transpose(self, *axes: Any) -> PrivArray:
        from .primitives import transpose

        if not axes:
            permutation = tuple(reversed(range(self.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], tuple):
            permutation = axes[0]
        else:
            permutation = tuple(axes)
        result: PrivArray = transpose(self, permutation)
        return result

    @property
    def T(self) -> PrivArray:
        return self.transpose()

    def sum(
        self,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
    ) -> PrivArray | SensitiveArray:
        from .primitives import reduce_sum

        if keepdims:
            raise NotImplementedError("keepdims is not supported by the prototype.")
        if axis is None:
            axes = tuple(range(self.ndim))
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = axis
        result: PrivArray | SensitiveArray = reduce_sum(self, axes)
        return result

    def __str__(self) -> str:
        return "<*** (jax.Array)>"

    def __repr__(self) -> str:
        return "<*** (jax.Array)>"


@egrpc.remoteclass
class SensitiveArray(Prisoner[jax.Array]):
    _norm_type: str

    def __init__(
        self,
        value: Any,
        distance: RealExpr,
        norm_type: str = "l1",
        *,
        parents: Sequence[Prisoner[Any]] = (),
        accountant: Accountant[Any] | None = None,
    ) -> None:
        self._norm_type = norm_type
        super().__init__(
            value=jnp.asarray(value),
            distance=distance,
            parents=parents,
            accountant=accountant,
        )

    @egrpc.property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._value.shape)

    @egrpc.property
    def ndim(self) -> int:
        return int(self._value.ndim)

    @egrpc.property
    def dtype(self) -> str:
        return str(self._value.dtype)

    @egrpc.property
    def weak_type(self) -> bool:
        return bool(getattr(self._value, "weak_type", False))

    @egrpc.property
    def norm_type(self) -> str:
        return self._norm_type

    def __str__(self) -> str:
        return "<*** (sensitive jax.Array)>"

    def __repr__(self) -> str:
        return "<*** (sensitive jax.Array)>"


@egrpc.function
def _asarray_from_numpy(
    array: PrivNDArray,
    dtype: str | None,
) -> PrivArray:
    return PrivArray(
        value=jnp.asarray(
            array._value,
            dtype=None if dtype is None else jnp.dtype(dtype),
        ),
        distance=array._distance,
        privacy_axis=array._privacy_axis,
        domain=ArrayDomain(
            norm_type=array._domain.norm_type,
            norm_bound=(
                None
                if array._domain.norm_bound is None
                else RealExpr(array._domain.norm_bound)
            ),
            value_range=array._domain.value_range,
        ),
        parents=[array],
        keep_alignment=True,
    )


@overload
def asarray(array: PrivNDArray, dtype: Any | None = None) -> PrivArray: ...


@overload
def asarray(array: Any, dtype: Any | None = None) -> jax.Array: ...


def asarray(
    array: Any,
    dtype: Any | None = None,
) -> PrivArray | jax.Array:
    if isinstance(array, PrivNDArray):
        dtype_name = None if dtype is None else str(jnp.dtype(dtype))
        return _asarray_from_numpy(array, dtype_name)
    return jnp.asarray(array, dtype=dtype)


__all__ = [
    "PrivArray",
    "SensitiveArray",
    "asarray",
]
