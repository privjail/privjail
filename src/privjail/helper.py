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
from typing import Any, overload

import egrpc
import jax

from .util import realnum, DPError, _secure_poisson_mask
from .numpy import PrivNDArray
from .alignment import new_alignment_signature
from .jax.array import PrivArray
from .jax.helper import _sample_jax_impl
from .jax.util import require_static

@overload
def clip_norm(arr: PrivNDArray, bound: realnum, ord: int | None = None) -> PrivNDArray: ...
@overload
def clip_norm(arr: PrivArray, bound: realnum, ord: int | None = None) -> PrivArray: ...
@overload
def clip_norm(arr: jax.Array, bound: realnum, ord: int | None = None) -> jax.Array: ...
@overload
def clip_norm(arr: Any, bound: realnum, ord: int | None = None) -> Any: ...

def clip_norm(arr: Any, bound: realnum, ord: int | None = None) -> Any:
    if isinstance(arr, PrivNDArray):
        from .numpy.helper import clip_norm as numpy_clip_norm
        return numpy_clip_norm(arr, bound, ord)
    require_static(bound, "clip_norm bound")
    require_static(ord, "clip_norm ord")
    import jax
    import jax.numpy as jnp
    from .jax.helper import clip_norm as jax_clip_norm
    leaves, treedef = jax.tree.flatten(arr)
    if not leaves:
        return arr
    arrays = [
        leaf if isinstance(leaf, PrivArray) else jnp.asarray(leaf)
        for leaf in leaves
    ]
    clipped = jax_clip_norm(arrays, bound, ord)
    return jax.tree.unflatten(treedef, clipped)

@overload
def normalize(arr: PrivNDArray, ord: int | None = None) -> PrivNDArray: ...
@overload
def normalize(arr: PrivArray, ord: int | None = None) -> PrivArray: ...
@overload
def normalize(arr: jax.Array, ord: int | None = None) -> jax.Array: ...

def normalize(arr: Any, ord: int | None = None) -> Any:
    if isinstance(arr, PrivNDArray):
        from .numpy.helper import normalize as numpy_normalize
        return numpy_normalize(arr, ord)
    require_static(ord, "normalize ord")
    import jax.numpy as jnp
    from .jax.helper import normalize as jax_normalize
    array = arr if isinstance(arr, PrivArray) else jnp.asarray(arr)
    return jax_normalize(array, ord)

@overload
def sample(array: PrivNDArray, /, *, q: float, method: str = "poisson") -> PrivNDArray: ...
@overload
def sample(*arrays: PrivNDArray, q: float, method: str = "poisson") -> tuple[PrivNDArray, ...]: ...
@overload
def sample(array: PrivArray, /, *, q: float, method: str = "poisson") -> PrivArray: ...
@overload
def sample(*arrays: PrivArray, q: float, method: str = "poisson") -> tuple[PrivArray, ...]: ...

def sample(
    *arrays: PrivNDArray | PrivArray,
    q: float,
    method: str = "poisson",
) -> PrivNDArray | PrivArray | tuple[PrivNDArray, ...] | tuple[PrivArray, ...]:
    if not arrays:
        raise ValueError("At least one array is required.")
    if all(isinstance(array, PrivNDArray) for array in arrays):
        numpy_result = _sample_impl(
            tuple(array for array in arrays if isinstance(array, PrivNDArray)),
            q,
            method,
        )
        if len(arrays) == 1:
            return numpy_result[0]
        return numpy_result
    elif all(isinstance(array, PrivArray) for array in arrays):
        jax_result = _sample_jax_impl(
            tuple(array for array in arrays if isinstance(array, PrivArray)),
            q,
            method,
        )
        if len(arrays) == 1:
            return jax_result[0]
        return jax_result
    raise TypeError("sample operands must use the same array backend.")

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
    mask = _secure_poisson_mask(n, q)

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

@egrpc.function
def shutdown_remote_server() -> None:
    egrpc.shutdown_server()

