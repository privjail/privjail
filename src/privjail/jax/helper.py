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

from collections.abc import Callable, Sequence
from functools import wraps
import math
from typing import Any

import egrpc
import jax
import jax.extend.core as jax_core
import jax.numpy as jnp

from ..alignment import assert_privacy_axis, new_alignment_signature
from ..realexpr import RealExpr
from ..util import DPError, _secure_poisson_mask
from .array import PrivArray
from .domain import ArrayDomain
from .util import assert_same_distance_and_accountant, require_static


def _normalize_abstract(
    aval: Any,
    *,
    ord: int,
) -> Any:
    del ord
    return aval


_normalize_p = jax_core.Primitive("normalize")
_normalize_p.def_abstract_eval(  # type: ignore[no-untyped-call]
    _normalize_abstract
)


def _normalize_array(
    array: jax.Array,
    ord: int,
) -> jax.Array:
    if array.ndim == 0:
        raise ValueError("normalize requires an array with at least one dimension.")
    array = jnp.where(jnp.isfinite(array), array, jnp.zeros_like(array))
    if array.size == 0:
        return array
    rows = array.reshape((array.shape[0], -1))
    norms = jnp.linalg.norm(rows, ord=ord, axis=1)
    normalized = array / (
        norms.reshape((array.shape[0],) + (1,) * (array.ndim - 1))
        + jnp.asarray(1e-12, dtype=array.dtype)
    )
    return jnp.where(
        jnp.isfinite(normalized),
        normalized,
        jnp.zeros_like(normalized),
    )


@egrpc.multifunction
def normalize(
    array: PrivArray,
    ord: int | None = None,
) -> PrivArray:
    ord = 2 if ord is None else ord
    if ord not in (1, 2):
        raise ValueError("normalize ord must be 1 or 2.")
    if array._privacy_axis != 0:
        raise DPError("normalize requires the privacy dimension at axis 0.")
    if not jnp.issubdtype(array._value.dtype, jnp.floating):
        raise TypeError("normalize requires a floating array.")

    return PrivArray(
        value=_normalize_array(array._value, ord),
        distance=array._distance,
        privacy_axis=0,
        domain=ArrayDomain(
            norm_type="l1" if ord == 1 else "l2",
            norm_bound=RealExpr(1.0),
            value_range=(-1.0, 1.0),
        ),
        parents=[array],
        keep_alignment=True,
    )


@normalize.register(remote=False)
def _(
    array: jax.Array,
    ord: int | None = None,
) -> jax.Array:
    static_ord = 2 if ord is None else ord
    if isinstance(array, jax.core.Tracer):
        result: jax.Array = _normalize_p.bind(  # type: ignore[no-untyped-call]
            array,
            ord=static_ord,
        )
        return result
    return _normalize_array(array, static_ord)


def _clip_norm_abstract(
    *avals: Any,
    bound: float,
    ord: int,
) -> list[Any]:
    del bound, ord
    return list(avals)


_clip_norm_p = jax_core.Primitive("clip_norm")
_clip_norm_p.multiple_results = True
_clip_norm_p.def_abstract_eval(  # type: ignore[no-untyped-call]
    _clip_norm_abstract
)


def _clip_norm_arrays(
    arrays: list[jax.Array],
    bound: float,
    ord: int,
) -> list[jax.Array]:
    if not arrays:
        return []
    if ord != 2:
        raise ValueError("JAX clip_norm currently supports only ord=2.")
    finite_arrays = [
        jnp.where(jnp.isfinite(array), array, jnp.zeros_like(array))
        for array in arrays
    ]
    squared_norms = jnp.zeros(
        (finite_arrays[0].shape[0],),
        dtype=finite_arrays[0].dtype,
    )
    for array in finite_arrays:
        squared_norms = squared_norms + jnp.sum(
            jnp.square(array),
            axis=tuple(range(1, array.ndim)),
        )
    norms = jnp.sqrt(squared_norms)
    bound_array = jnp.asarray(bound, dtype=norms.dtype)
    scales = bound_array / jnp.maximum(norms, bound_array)
    clipped_arrays: list[jax.Array] = []
    for array in finite_arrays:
        clipped = array * scales.reshape(
            (array.shape[0],) + (1,) * (array.ndim - 1),
        )
        clipped_arrays.append(
            jnp.where(
                jnp.isfinite(clipped),
                clipped,
                jnp.zeros_like(clipped),
            )
        )
    return clipped_arrays


@egrpc.multifunction
def clip_norm(
    arrays: list[PrivArray],
    bound: float,
    ord: int | None = None,
) -> list[PrivArray]:
    if not arrays:
        return []
    static_ord = 2 if ord is None else ord
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("clip_norm bound must be finite and > 0.")
    assert_privacy_axis(*arrays)
    assert_same_distance_and_accountant(*arrays)
    if any(array._privacy_axis != 0 for array in arrays):
        raise DPError("clip_norm requires the privacy dimension at axis 0.")
    if any(array._value.ndim < 1 for array in arrays):
        raise ValueError("Clipped gradient leaves must have a batch dimension.")
    if any(array._value.shape[0] != arrays[0]._value.shape[0] for array in arrays[1:]):
        raise ValueError("Clipped gradient leaves must have the same batch dimension.")
    if any(array._value.dtype != arrays[0]._value.dtype for array in arrays[1:]):
        raise TypeError("Clipped gradient leaves must have the same dtype.")
    if not jnp.issubdtype(arrays[0]._value.dtype, jnp.floating):
        raise TypeError("Clipped gradient leaves must have a floating dtype.")

    clipped_arrays = _clip_norm_arrays(
        [array._value for array in arrays],
        bound,
        static_ord,
    )
    norm_components = RealExpr(bound).create_l2_components(len(arrays))
    results: list[PrivArray] = []
    for array, clipped, norm_component in zip(
        arrays,
        clipped_arrays,
        norm_components,
        strict=True,
    ):
        results.append(
            PrivArray(
                value=clipped,
                distance=arrays[0]._distance,
                privacy_axis=0,
                domain=ArrayDomain(
                    norm_type="l2",
                    norm_bound=norm_component,
                ),
                parents=arrays,
                keep_alignment=True,
            )
        )
    return results


@clip_norm.register(remote=False)
def _(
    arrays: list[jax.Array],
    bound: float,
    ord: int | None = None,
) -> list[jax.Array]:
    static_ord = 2 if ord is None else ord
    if any(isinstance(array, jax.core.Tracer) for array in arrays):
        return _clip_norm_p.bind(  # type: ignore[no-untyped-call,no-any-return]
            *arrays,
            bound=bound,
            ord=static_ord,
        )
    return _clip_norm_arrays(arrays, bound, static_ord)


@egrpc.function
def _sample_jax_impl(
    arrays: tuple[PrivArray, ...],
    q: float,
    method: str,
) -> tuple[PrivArray, ...]:
    if not arrays:
        raise ValueError("At least one array is required.")
    if not 0.0 < q <= 1.0:
        raise ValueError("Sampling rate q must be in (0, 1].")
    if method != "poisson":
        raise ValueError(f"Unknown sampling method: {method!r}")

    first = arrays[0]
    if not all(
        array._alignment_signature == first._alignment_signature
        for array in arrays[1:]
    ):
        raise DPError("All arrays must have the same alignment_signature.")
    if any(array._privacy_axis != 0 for array in arrays):
        raise DPError("JAX subsampling requires the privacy dimension at axis 0.")
    if any(
        array._value.shape[0] != first._value.shape[0]
        for array in arrays[1:]
    ):
        raise DPError("Aligned sampling inputs must have the same record count.")
    if any(
        array._accountant is not first._accountant
        for array in arrays[1:]
    ):
        raise DPError("All sampling inputs must share one accountant.")

    if any(float(array._distance.max()) != 1.0 for array in arrays):
        raise DPError("Subsampling requires adjacent databases (max_distance=1)")

    mask = _secure_poisson_mask(first._value.shape[0], q)
    indices = jnp.asarray(mask.nonzero()[0])
    child_accountant = first._accountant.create_subsampling_accountant(q)
    signature = new_alignment_signature()

    results: list[PrivArray] = []
    for array in arrays:
        result = PrivArray(
            value=jnp.take(array._value, indices, axis=0),
            distance=array._distance,
            privacy_axis=0,
            domain=array._domain,
            parents=[array],
            accountant=child_accountant,
            keep_alignment=False,
        )
        result._alignment_signature = signature
        results.append(result)
    return tuple(results)


def clipped_grad(
    fun: Callable[..., jax.Array],
    argnums: int | Sequence[int] = 0,
    *,
    l2_clip_norm: float,
    batch_argnums: int | Sequence[int] = 1,
) -> Callable[..., Any]:
    """Transform a scalar batch loss into a clipped gradient-sum function.

    Each argument selected by ``batch_argnums`` is sliced along axis 0. The
    slices are given a singleton batch dimension before calling ``fun``, so a
    conventional batch loss remains valid but reductions such as ``mean`` are
    evaluated with batch size one. The resulting per-record gradients are
    globally clipped together and summed without normalization.
    """

    require_static(l2_clip_norm, "clip_norm bound")
    batch_indices = (
        (batch_argnums,) if isinstance(batch_argnums, int) else tuple(batch_argnums)
    )
    if not batch_indices:
        raise ValueError("batch_argnums must select at least one argument.")
    gradient = jax.grad(fun, argnums=argnums)

    @wraps(fun)
    def clipped_gradient_sum(*args: Any, **kwargs: Any) -> Any:
        argument_count = len(args)
        normalized_indices = tuple(
            index + argument_count if index < 0 else index
            for index in batch_indices
        )
        if any(not 0 <= index < argument_count for index in normalized_indices):
            raise ValueError("batch_argnums contains an out-of-range argument index.")
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError("batch_argnums must not contain duplicate indices.")

        batched_values = tuple(args[index] for index in normalized_indices)

        def gradient_for_record(*record_values: Any) -> Any:
            singleton_args = list(args)
            for index, record_value in zip(
                normalized_indices,
                record_values,
                strict=True,
            ):
                singleton_args[index] = jax.tree.map(
                    lambda leaf: jnp.expand_dims(leaf, axis=0),
                    record_value,
                )
            return gradient(*singleton_args, **kwargs)

        per_record_gradients = jax.vmap(gradient_for_record)(*batched_values)
        leaves, treedef = jax.tree.flatten(per_record_gradients)
        clipped_leaves = clip_norm(
            [jnp.asarray(leaf) for leaf in leaves],
            l2_clip_norm,
            None,
        )
        clipped = jax.tree.unflatten(treedef, clipped_leaves)
        return jax.tree.map(lambda leaf: leaf.sum(axis=0), clipped)

    return clipped_gradient_sum


__all__ = [
    "clip_norm",
    "clipped_grad",
    "normalize",
]
