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

import uuid
from collections.abc import Sequence
from typing import Any, cast

import egrpc
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import privjail as pj
import privjail.jax as pjx
import privjail.numpy as pnp
from privjail.jax.call_batch import (
    pack_outputs,
    validate_call_batch,
)
from privjail.jax.mechanism import (
    gaussian_mechanism as trusted_gaussian_mechanism,
)
from privjail.realexpr import joint_l2_max


def _private_array(
    value: Any,
    *,
    privacy_axis: int = 0,
    value_range: tuple[float | None, float | None] | None = None,
    accountant: pj.Accountant[Any] | None = None,
) -> pjx.PrivArray:
    if accountant is None:
        accountant = pj.ApproxDPAccountant()
        accountant.set_as_root(name=f"jax-test-{uuid.uuid4()}")
    numpy_array = pnp.PrivNDArray(
        np.asarray(value, dtype=np.float32),
        distance=pj.RealExpr(1),
        privacy_axis=privacy_axis,
        domain=pnp.NDArrayDomain(value_range=value_range),
        accountant=accountant,
    )
    result = pjx.asarray(numpy_array)
    assert isinstance(result, pjx.PrivArray)
    return result


def _host(value: jax.Array) -> np.ndarray[Any, Any]:
    return np.asarray(jax.device_get(value))


def _aligned_private_array(
    value: Any,
    parent: pjx.PrivArray,
) -> pjx.PrivArray:
    return pjx.PrivArray(
        np.asarray(value, dtype=np.float32),
        distance=parent._distance,
        privacy_axis=parent._privacy_axis,
        parents=[parent],
        keep_alignment=True,
    )


def _mnist_loss_one(
    params: dict[str, jax.Array],
    x_one: jax.Array,
    y_one: jax.Array,
) -> jax.Array:
    hidden = jax.nn.relu(x_one @ params["w1"])
    log_probs = jax.nn.log_softmax(hidden @ params["w2"])
    return -log_probs[y_one]


def _mnist_batch_loss(
    params: dict[str, jax.Array],
    x_batch: jax.Array,
    y_batch: jax.Array,
) -> jax.Array:
    losses = jax.vmap(_mnist_loss_one, in_axes=(None, 0, 0))(
        params,
        x_batch,
        y_batch,
    )
    return losses.mean()


_mnist_per_example_gradients = jax.vmap(
    jax.grad(_mnist_loss_one),
    in_axes=(None, 0, 0),
)


def test_asarray_converts_dtype_before_jax_preprocessing() -> None:
    accountant = pj.ApproxDPAccountant()
    accountant.set_as_root(name=f"jax-asarray-{uuid.uuid4()}")
    numpy_private = pnp.PrivNDArray(
        np.arange(6, dtype=np.uint8).reshape(3, 2),
        distance=pj.RealExpr(1),
        privacy_axis=0,
        accountant=accountant,
    )

    private = pjx.asarray(numpy_private, dtype=jnp.float32)
    assert isinstance(private, pjx.PrivArray)
    preprocessed = private.reshape((-1, 2)) / 255.0

    assert isinstance(preprocessed, pjx.PrivArray)
    assert preprocessed._value.dtype == jnp.float32
    np.testing.assert_allclose(
        _host(preprocessed._value),
        np.arange(6, dtype=np.float32).reshape(3, 2) / 255.0,
    )


def test_sample_accepts_aligned_jax_private_arrays() -> None:
    private = _private_array(np.arange(8).reshape(4, 2))
    labels = _aligned_private_array(np.arange(4), private)

    sampled_private, sampled_labels = pj.sample(
        private,
        labels,
        q=1.0,
    )

    assert isinstance(sampled_private, pjx.PrivArray)
    assert isinstance(sampled_labels, pjx.PrivArray)
    np.testing.assert_allclose(
        _host(sampled_private._value),
        _host(private._value),
    )
    np.testing.assert_allclose(
        _host(sampled_labels._value),
        _host(labels._value),
    )
    assert (
        sampled_private._alignment_signature
        == sampled_labels._alignment_signature
    )
    assert sampled_private._alignment_signature != private._alignment_signature
    assert sampled_private._accountant is sampled_labels._accountant
    assert sampled_private._accountant is not private._accountant


def test_jax_sampled_release_uses_its_subsampling_accountant() -> None:
    root = pj.RDPAccountant(alpha=[2.0])
    root.set_as_root(name=f"jax-sample-accounting-{uuid.uuid4()}")
    private = _private_array(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        accountant=root,
    )
    sampled = pj.sample(private, q=0.5)
    assert isinstance(sampled, pjx.PrivArray)

    @pjx.jit
    def release(x: jax.Array) -> jax.Array:
        sensitive = pj.clip_norm(x, 1.0).sum(axis=0)
        return cast(
            jax.Array,
            pj.gaussian_mechanism(sensitive, scale=2.0),
        )

    release(sampled)

    assert sampled._accountant.budget_spent[2.0] > 0
    assert 0 < root.budget_spent[2.0] < 0.25


def _conv3d_parameters() -> dict[str, jax.Array]:
    return {
        "kernel": (
            jnp.arange(54, dtype=jnp.float32).reshape(3, 3, 3, 1, 2)
            / 100.0
        ),
        "dense": (
            jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 10.0
        ),
    }


def _conv3d_logits(
    params: dict[str, jax.Array],
    x_one: jax.Array,
) -> jax.Array:
    convolved = jax.lax.conv_general_dilated(
        x_one[None, ...],
        params["kernel"],
        window_strides=(1, 1, 1),
        padding="SAME",
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )
    pooled = jnp.mean(
        jax.nn.relu(convolved),
        axis=(1, 2, 3),
    )[0]
    return pooled @ params["dense"]


def _conv3d_loss_one(
    params: dict[str, jax.Array],
    x_one: jax.Array,
    y_one: jax.Array,
) -> jax.Array:
    return -jax.nn.log_softmax(_conv3d_logits(params, x_one))[y_one]


_conv3d_per_example_gradients = jax.vmap(
    jax.grad(_conv3d_loss_one),
    in_axes=(None, 0, 0),
)


def test_private_shape_hides_private_dimension() -> None:
    first = _private_array(np.zeros((4, 3)))
    second = _private_array(np.zeros((9, 3)))

    first_shape = first.shape
    second_shape = second.shape

    assert isinstance(first_shape[0], pj.SensitiveDimInt)
    assert isinstance(second_shape[0], pj.SensitiveDimInt)
    assert first_shape[1:] == second_shape[1:] == (3,)


def test_jit_supports_nested_jit_and_custom_jvp() -> None:
    @jax.jit
    def jitted_shift(x: jax.Array) -> jax.Array:
        return x + 1.0

    @jax.custom_jvp
    def custom_shift(x: jax.Array) -> jax.Array:
        return cast(jax.Array, jitted_shift(x))

    @custom_shift.defjvp
    def custom_shift_jvp(
        primals: tuple[jax.Array],
        tangents: tuple[jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        (x,) = primals
        (tangent,) = tangents
        return custom_shift(x), tangent

    @pjx.jit
    def transform(x: jax.Array) -> jax.Array:
        return custom_shift(x) * 2.0

    private = _private_array(np.ones((3,), dtype=np.float32))
    result = transform(private)

    assert isinstance(result, pjx.PrivArray)
    np.testing.assert_allclose(_host(result._value), [4.0, 4.0, 4.0])
    assert result._alignment_signature == private._alignment_signature


def test_jit_binds_output_alignment_to_each_input() -> None:
    @pjx.jit
    def pair(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return x * 2.0, x + 1.0

    first = _private_array(np.arange(6).reshape(2, 3))
    second = _private_array(np.arange(9).reshape(3, 3))

    first_pair = pair(first)
    second_pair = pair(second)
    assert all(isinstance(value, pjx.PrivArray) for value in first_pair)
    assert all(isinstance(value, pjx.PrivArray) for value in second_pair)
    first_private_pair = cast(
        tuple[pjx.PrivArray, pjx.PrivArray],
        first_pair,
    )
    second_private_pair = cast(
        tuple[pjx.PrivArray, pjx.PrivArray],
        second_pair,
    )

    assert all(
        value._alignment_signature == first._alignment_signature
        for value in first_private_pair
    )
    assert all(
        value._alignment_signature == second._alignment_signature
        for value in second_private_pair
    )
    assert (
        first_private_pair[0]._alignment_signature
        != second_private_pair[0]._alignment_signature
    )


def test_aligned_operands_require_equal_distances() -> None:
    first = _private_array(np.arange(6).reshape(2, 3))
    second = pjx.PrivArray(
        np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        distance=first._distance * 2,
        privacy_axis=0,
        parents=[first],
        keep_alignment=True,
    )

    with pytest.raises(pj.DPError, match="distance expressions"):
        first + second
    with pytest.raises(pj.DPError, match="distance expressions"):
        pj.clip_norm((first, second), 1.0)


def test_jax_grad_and_vmap_produce_private_per_example_gradients() -> None:
    def loss_one(
        params: dict[str, jax.Array],
        x_one: jax.Array,
        y_one: jax.Array,
    ) -> jax.Array:
        difference = x_one @ params["w"] - y_one
        return jnp.sum(difference * difference)

    gradient_batch = jax.vmap(
        jax.grad(loss_one),
        in_axes=(None, 0, 0),
    )
    wrapped = pjx.jit(gradient_batch)

    params = {
        "w": jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10.0,
    }
    private_x = _private_array(np.arange(12).reshape(4, 3))
    private_y = _aligned_private_array(np.arange(8).reshape(4, 2), private_x)

    result = wrapped(params, private_x, private_y)
    raw_result = gradient_batch(
        params,
        private_x._value,
        private_y._value,
    )

    assert isinstance(result["w"], pjx.PrivArray)
    np.testing.assert_allclose(
        _host(result["w"]._value),
        _host(raw_result["w"]),
    )
    assert result["w"]._value.shape == (4, 3, 2)
    assert result["w"]._privacy_axis == 0
    assert result["w"]._alignment_signature == private_x._alignment_signature


def test_grad_vmap_clip_and_sum_matches_raw_jax() -> None:
    def loss_one(
        params: dict[str, jax.Array],
        x_one: jax.Array,
        y_one: jax.Array,
    ) -> jax.Array:
        difference = x_one @ params["w"] + params["b"] - y_one
        return jnp.sum(difference * difference)

    gradient_batch = jax.vmap(
        jax.grad(loss_one),
        in_axes=(None, 0, 0),
    )

    def clipped_gradient_sum(
        params: dict[str, jax.Array],
        x_batch: jax.Array,
        y_batch: jax.Array,
        clip_bound: float,
    ) -> dict[str, jax.Array]:
        per_example = gradient_batch(params, x_batch, y_batch)
        clipped = pj.clip_norm(per_example, clip_bound)
        return cast(
            dict[str, jax.Array],
            jax.tree.map(lambda leaf: leaf.sum(axis=0), clipped),
        )

    wrapped = pjx.jit(
        clipped_gradient_sum,
        static_argnames="clip_bound",
    )
    params = {
        "w": jnp.arange(6, dtype=jnp.float32).reshape(3, 2) / 10.0,
        "b": jnp.zeros((2,), dtype=jnp.float32),
    }
    private_x = _private_array(np.arange(12).reshape(4, 3))
    private_y = _aligned_private_array(np.arange(8).reshape(4, 2), private_x)

    result = wrapped(params, private_x, private_y, clip_bound=1.5)
    raw_result = clipped_gradient_sum(
        params,
        private_x._value,
        private_y._value,
        1.5,
    )

    assert all(isinstance(leaf, pjx.SensitiveArray) for leaf in result.values())
    protected_result = cast(dict[str, pjx.SensitiveArray], result)
    for name in params:
        np.testing.assert_allclose(
            _host(protected_result[name]._value),
            _host(raw_result[name]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert protected_result[name]._distance.max() == pytest.approx(1.5)
        assert protected_result[name]._norm_type == "l2"
    assert joint_l2_max(
        [protected_result["w"]._distance, protected_result["b"]._distance]
    ) == pytest.approx(1.5)


def test_clipped_grad_batch_mean_matches_explicit_mnist_gradients() -> None:
    def explicit_clipped_gradient_sum(
        params: dict[str, jax.Array],
        x_batch: jax.Array,
        y_batch: jax.Array,
        clip_bound: float,
    ) -> dict[str, jax.Array]:
        per_example = _mnist_per_example_gradients(
            params,
            x_batch,
            y_batch,
        )
        clipped = pj.clip_norm(per_example, clip_bound)
        return cast(
            dict[str, jax.Array],
            jax.tree.map(lambda leaf: leaf.sum(axis=0), clipped),
        )

    clipped_gradient_sum = pjx.clipped_grad(
        _mnist_batch_loss,
        l2_clip_norm=1.25,
        batch_argnums=(1, 2),
    )
    params = {
        "w1": jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 100.0,
        "w2": jnp.arange(40, dtype=jnp.float32).reshape(4, 10) / 100.0,
    }
    private_x = _private_array(
        np.arange(24, dtype=np.float32).reshape(3, 8) / 24.0
    )
    private_y = pjx.PrivArray(
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
        distance=private_x._distance,
        privacy_axis=0,
        parents=[private_x],
        keep_alignment=True,
    )
    wrapped = pjx.jit(clipped_gradient_sum)

    result = wrapped(params, private_x, private_y)
    raw_result = explicit_clipped_gradient_sum(
        params,
        private_x._value,
        private_y._value,
        1.25,
    )
    public_result = clipped_gradient_sum(
        params,
        private_x._value,
        private_y._value,
    )

    assert all(isinstance(leaf, pjx.SensitiveArray) for leaf in result.values())
    protected_result = cast(dict[str, pjx.SensitiveArray], result)
    for name in params:
        np.testing.assert_allclose(
            _host(public_result[name]),
            _host(raw_result[name]),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            _host(protected_result[name]._value),
            _host(raw_result[name]),
            rtol=1e-6,
            atol=1e-6,
        )
        assert protected_result[name]._distance.max() == pytest.approx(1.25)
    assert joint_l2_max(
        [
            protected_result["w1"]._distance,
            protected_result["w2"]._distance,
        ]
    ) == pytest.approx(1.25)


def test_clipped_grad_supports_an_empty_private_batch() -> None:
    params = {
        "w1": jnp.zeros((8, 4), dtype=jnp.float32),
        "w2": jnp.zeros((4, 10), dtype=jnp.float32),
    }
    private_x = _private_array(np.empty((0, 8), dtype=np.float32))
    private_y = pjx.PrivArray(
        jnp.empty((0,), dtype=jnp.int32),
        distance=private_x._distance,
        privacy_axis=0,
        parents=[private_x],
        keep_alignment=True,
    )
    clipped_gradient_sum = pjx.jit(
        pjx.clipped_grad(
            _mnist_batch_loss,
            l2_clip_norm=1.25,
            batch_argnums=(1, 2),
        )
    )

    result = clipped_gradient_sum(params, private_x, private_y)

    assert all(isinstance(leaf, pjx.SensitiveArray) for leaf in result.values())
    protected = cast(dict[str, pjx.SensitiveArray], result)
    for name in params:
        np.testing.assert_array_equal(
            _host(protected[name]._value),
            np.zeros(params[name].shape, dtype=np.float32),
        )
        assert protected[name]._distance.max() == pytest.approx(1.25)


def test_conv3d_grad_vmap_matches_jax() -> None:
    params = _conv3d_parameters()
    private_x = _private_array(
        np.arange(2 * 4 * 4 * 4, dtype=np.float32).reshape(
            2,
            4,
            4,
            4,
            1,
        )
        / 100.0
    )
    private_y = pjx.PrivArray(
        jnp.asarray([0, 2], dtype=jnp.int32),
        distance=private_x._distance,
        privacy_axis=0,
        parents=[private_x],
        keep_alignment=True,
    )
    wrapped = pjx.jit(_conv3d_per_example_gradients)

    result = wrapped(params, private_x, private_y)
    expected = _conv3d_per_example_gradients(
        params,
        private_x._value,
        private_y._value,
    )

    for protected, raw in zip(
        jax.tree.leaves(result),
        jax.tree.leaves(expected),
        strict=True,
    ):
        assert isinstance(protected, pjx.PrivArray)
        np.testing.assert_allclose(
            _host(protected._value),
            _host(raw),
            rtol=1e-6,
            atol=1e-6,
        )
        assert protected._privacy_axis == 0
        assert (
            protected._alignment_signature
            == private_x._alignment_signature
        )
        assert protected._distance.structurally_equal(private_x._distance)


def test_multilayer_conv3d_clipped_grad_matches_jax() -> None:
    params = {
        "conv1": (
            jnp.arange(54, dtype=jnp.float32).reshape(3, 3, 3, 1, 2)
            / 100.0
        ),
        "conv2": (
            jnp.arange(108, dtype=jnp.float32).reshape(3, 3, 3, 2, 2)
            / 100.0
        ),
        "dense": (
            jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 10.0
        ),
    }

    def loss_fn(
        current_params: dict[str, jax.Array],
        x_batch: jax.Array,
        y_batch: jax.Array,
    ) -> jax.Array:
        hidden = jax.nn.relu(
            jax.lax.conv_general_dilated(
                x_batch,
                current_params["conv1"],
                window_strides=(2, 2, 2),
                padding="SAME",
                dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
            )
        )
        hidden = jax.nn.relu(
            jax.lax.conv_general_dilated(
                hidden,
                current_params["conv2"],
                window_strides=(2, 2, 2),
                padding="SAME",
                dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
            )
        )
        pooled = jnp.mean(hidden, axis=(1, 2, 3))
        log_probs = jax.nn.log_softmax(
            pooled @ current_params["dense"]
        )
        losses = -jax.vmap(lambda row, label: row[label])(
            log_probs,
            y_batch,
        )
        return cast(jax.Array, losses.mean())

    clipped_gradient_sum = pjx.clipped_grad(
        loss_fn,
        l2_clip_norm=1.25,
        batch_argnums=(1, 2),
    )
    private_x = _private_array(
        np.arange(2 * 8 * 8 * 8, dtype=np.float32).reshape(
            2,
            8,
            8,
            8,
            1,
        )
        / 1000.0
    )
    private_y = pjx.PrivArray(
        jnp.asarray([0, 2], dtype=jnp.int32),
        distance=private_x._distance,
        privacy_axis=0,
        parents=[private_x],
        keep_alignment=True,
    )

    result = pjx.jit(clipped_gradient_sum)(
        params,
        private_x,
        private_y,
    )
    expected = clipped_gradient_sum(
        params,
        private_x._value,
        private_y._value,
    )

    for protected, raw in zip(
        jax.tree.leaves(result),
        jax.tree.leaves(expected),
        strict=True,
    ):
        assert isinstance(protected, pjx.SensitiveArray)
        np.testing.assert_allclose(
            _host(protected._value),
            _host(raw),
            rtol=1e-6,
            atol=1e-6,
        )
        assert protected._distance.max() == pytest.approx(1.25)


def test_conv3d_eager_lax_api_matches_jax() -> None:
    private = _private_array(
        np.arange(2 * 4 * 4 * 4, dtype=np.float32).reshape(
            2,
            4,
            4,
            4,
            1,
        )
        / 100.0
    )
    kernel = _conv3d_parameters()["kernel"]

    result = pjx.lax.conv_general_dilated(
        private,
        kernel,
        window_strides=(1, 1, 1),
        padding="SAME",
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )
    expected = jax.lax.conv_general_dilated(
        private._value,
        kernel,
        window_strides=(1, 1, 1),
        padding="SAME",
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )

    assert isinstance(result, pjx.PrivArray)
    np.testing.assert_allclose(
        _host(result._value),
        _host(expected),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result._privacy_axis == 0
    assert result._distance.max() == pytest.approx(1.0)
    assert result._alignment_signature == private._alignment_signature


def test_conv3d_dp_sgd_step_releases_and_updates_public_params() -> None:
    def dp_sgd_step(
        params: dict[str, jax.Array],
        x_batch: jax.Array,
        y_batch: jax.Array,
        clip_bound: float,
        noise_scale: float,
    ) -> dict[str, jax.Array]:
        per_example = _conv3d_per_example_gradients(
            params,
            x_batch,
            y_batch,
        )
        clipped = pj.clip_norm(per_example, clip_bound)
        gradient_sums = jax.tree.map(
            lambda leaf: leaf.sum(axis=0),
            clipped,
        )
        noisy_sums = pj.gaussian_mechanism(
            gradient_sums,
            scale=noise_scale,
        )
        return cast(
            dict[str, jax.Array],
            jax.tree.map(
                lambda parameter, gradient: parameter - 0.05 * gradient,
                params,
                noisy_sums,
            ),
        )

    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-conv3d-noise-{uuid.uuid4()}")
    private_x = _private_array(
        np.arange(2 * 4 * 4 * 4, dtype=np.float32).reshape(
            2,
            4,
            4,
            4,
            1,
        )
        / 100.0,
        accountant=accountant,
    )
    private_y = pjx.PrivArray(
        jnp.asarray([0, 2], dtype=jnp.int32),
        distance=private_x._distance,
        privacy_axis=0,
        parents=[private_x],
        keep_alignment=True,
    )
    params = _conv3d_parameters()
    wrapped = pjx.jit(
        dp_sgd_step,
        static_argnames=("clip_bound", "noise_scale"),
    )

    result = wrapped(
        params,
        private_x,
        private_y,
        clip_bound=1.0,
        noise_scale=2.0,
    )

    assert result.keys() == params.keys()
    for name in params:
        assert isinstance(result[name], jax.Array)
        assert result[name].shape == params[name].shape
        assert bool(jnp.all(jnp.isfinite(result[name])))
    assert accountant.budget_spent[2.0] > 0


def test_conv3d_grad_vmap_supports_an_empty_private_batch() -> None:
    params = _conv3d_parameters()
    private_x = _private_array(
        np.empty((0, 4, 4, 4, 1), dtype=np.float32)
    )
    private_y = pjx.PrivArray(
        jnp.empty((0,), dtype=jnp.int32),
        distance=private_x._distance,
        privacy_axis=0,
        parents=[private_x],
        keep_alignment=True,
    )

    result = pjx.jit(_conv3d_per_example_gradients)(
        params,
        private_x,
        private_y,
    )
    expected = _conv3d_per_example_gradients(
        params,
        private_x._value,
        private_y._value,
    )

    for protected, raw in zip(
        jax.tree.leaves(result),
        jax.tree.leaves(expected),
        strict=True,
    ):
        assert isinstance(protected, pjx.PrivArray)
        np.testing.assert_array_equal(
            _host(protected._value),
            _host(raw),
        )
        assert (
            protected._alignment_signature
            == private_x._alignment_signature
        )


def test_conv3d_rejects_a_private_spatial_dimension() -> None:
    private = _private_array(
        np.ones((1, 4, 4, 4, 1), dtype=np.float32),
        privacy_axis=1,
    )
    kernel = _conv3d_parameters()["kernel"]

    @pjx.jit
    def convolve(x: jax.Array) -> jax.Array:
        return jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1, 1),
            padding="SAME",
            dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
        )

    with pytest.raises(
        pj.DPError,
        match="batch dimension as the privacy dimension",
    ):
        convolve(private)


def test_conv3d_rejects_a_non_recordwise_group_count() -> None:
    lhs = _private_array(
        np.ones((2, 4, 4, 4, 1), dtype=np.float32)
    )
    rhs = pjx.PrivArray(
        jnp.ones((1, 4, 4, 4, 4), dtype=jnp.float32),
        distance=lhs._distance * 2,
        privacy_axis=4,
        parents=[lhs],
        keep_alignment=True,
    )

    with pytest.raises(
        pj.DPError,
        match="record-aligned grouped kernel gradient",
    ):
        pjx.lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides=(1, 1, 1),
            padding=((1, 1), (1, 1), (1, 1)),
            dimension_numbers=("CDHWN", "IDHWO", "DHWNC"),
            feature_group_count=1,
        )


def test_rev_preserves_alignment_outside_the_privacy_dimension() -> None:
    private = _private_array(
        np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    )

    @pjx.jit
    def reverse_spatial_axes(x: jax.Array) -> jax.Array:
        return jax.lax.rev(x, (1, 2))

    result = reverse_spatial_axes(private)

    assert isinstance(result, pjx.PrivArray)
    np.testing.assert_array_equal(
        _host(result._value),
        _host(jax.lax.rev(private._value, (1, 2))),
    )
    assert result._alignment_signature == private._alignment_signature
    assert result._distance.structurally_equal(private._distance)


def test_rev_rejects_the_privacy_dimension() -> None:
    private = _private_array(np.arange(6).reshape(2, 3))

    @pjx.jit
    def reverse_records(x: jax.Array) -> jax.Array:
        return jax.lax.rev(x, (0,))

    with pytest.raises(pj.DPError, match="cannot reverse"):
        reverse_records(private)


def test_mnist_mlp_dp_sgd_step_releases_once_and_updates_public_params() -> None:
    def dp_sgd_step(
        params: dict[str, jax.Array],
        x_batch: jax.Array,
        y_batch: jax.Array,
        clip_bound: float,
        noise_multiplier: float,
        expected_batch_size: float,
        learning_rate: float,
    ) -> dict[str, jax.Array]:
        gradient_sums = pjx.clipped_grad(
            _mnist_batch_loss,
            l2_clip_norm=clip_bound,
            batch_argnums=(1, 2),
        )(params, x_batch, y_batch)
        noisy_sums = pj.gaussian_mechanism(
            gradient_sums,
            scale=noise_multiplier * clip_bound,
        )
        gradients = jax.tree.map(
            lambda leaf: leaf / expected_batch_size,
            noisy_sums,
        )
        return cast(
            dict[str, jax.Array],
            jax.tree.map(
                lambda parameter, gradient: (
                    parameter - learning_rate * gradient
                ),
                params,
                gradients,
            ),
        )

    accountant = pj.RDPAccountant(alpha=[2.0, 3.0, 4.0])
    accountant.set_as_root(name=f"jax-noise-{uuid.uuid4()}")
    private_x = _private_array(
        np.arange(24, dtype=np.float32).reshape(3, 8) / 24.0,
        accountant=accountant,
    )
    private_y = pjx.PrivArray(
        jnp.asarray([1, 2, 3], dtype=jnp.int32),
        distance=private_x._distance,
        privacy_axis=0,
        parents=[private_x],
        keep_alignment=True,
    )
    params = {
        "w1": jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 100.0,
        "w2": jnp.arange(40, dtype=jnp.float32).reshape(4, 10) / 100.0,
    }
    wrapped = pjx.jit(
        dp_sgd_step,
        static_argnames=("clip_bound", "noise_multiplier"),
    )

    first = wrapped(
        params,
        private_x,
        private_y,
        clip_bound=1.25,
        noise_multiplier=2.0,
        expected_batch_size=3.0,
        learning_rate=0.1,
    )
    np.testing.assert_allclose(
        list(accountant.budget_spent.values()),
        [0.25, 0.375, 0.5],
    )
    second = wrapped(
        params,
        private_x,
        private_y,
        clip_bound=1.25,
        noise_multiplier=2.0,
        expected_batch_size=3.0,
        learning_rate=0.1,
    )

    assert all(isinstance(leaf, jax.Array) for leaf in first.values())
    assert first["w1"].shape == params["w1"].shape
    assert first["w2"].shape == params["w2"].shape
    assert bool(jnp.any(first["w1"] != second["w1"]))
    np.testing.assert_allclose(
        list(accountant.budget_spent.values()),
        [0.5, 0.75, 1.0],
    )


def test_gaussian_release_can_return_sensitive_and_public_outputs() -> None:
    def release(
        x: jax.Array,
        clip_bound: float,
        noise_scale: float,
    ) -> tuple[jax.Array, jax.Array]:
        clipped = pj.clip_norm(x, clip_bound)
        sensitive = clipped.sum(axis=0)
        noisy = pj.gaussian_mechanism(
            sensitive,
            scale=noise_scale,
        )
        return sensitive, noisy

    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-mixed-{uuid.uuid4()}")
    private = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=accountant,
    )
    wrapped = pjx.jit(
        release,
        static_argnames=("clip_bound", "noise_scale"),
    )

    sensitive, public = wrapped(
        private,
        clip_bound=1.0,
        noise_scale=2.0,
    )

    assert isinstance(sensitive, pjx.SensitiveArray)
    assert isinstance(public, jax.Array)
    assert sensitive.shape == (3,)
    assert sensitive.dtype == "float32"
    assert not sensitive.weak_type
    assert public.shape == (3,)
    assert accountant.budget_spent == {2.0: pytest.approx(0.25)}


def test_gaussian_scale_argument_must_be_static() -> None:
    def release(
        x: jax.Array,
        scale: float,
    ) -> jax.Array:
        clipped = pj.clip_norm(x, 1.0)
        sensitive = clipped.sum(axis=0)
        return cast(
            jax.Array,
            pj.gaussian_mechanism(sensitive, scale=scale),
        )

    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-static-noise-{uuid.uuid4()}")
    private = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=accountant,
    )

    with pytest.raises(TypeError, match="scale must be static"):
        pjx.jit(release)(private, 2.0)


def test_eager_gaussian_uses_the_same_trusted_release_rule() -> None:
    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-eager-noise-{uuid.uuid4()}")
    private = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=accountant,
    )
    clipped = pj.clip_norm(private, 1.0)
    sensitive = clipped.sum(axis=0)
    assert isinstance(sensitive, pjx.SensitiveArray)

    result = pj.gaussian_mechanism(sensitive, scale=2.0)

    assert isinstance(result, jax.Array)
    assert result.shape == (3,)
    assert accountant.budget_spent == {2.0: pytest.approx(0.25)}


def test_global_clip_stores_the_joint_bound_in_array_domains() -> None:
    first = _private_array(np.arange(6).reshape(2, 3))
    second = _aligned_private_array(
        np.arange(6, 12).reshape(2, 3),
        first,
    )

    clipped = pj.clip_norm(
        {"first": first, "second": second},
        1.5,
    )

    first_bound = clipped["first"]._domain.norm_bound
    second_bound = clipped["second"]._domain.norm_bound
    assert first_bound is not None
    assert second_bound is not None
    assert joint_l2_max([first_bound, second_bound]) == pytest.approx(1.5)
    assert clipped["first"].domain.norm_bound == pytest.approx(1.5)
    assert clipped["second"].domain.norm_bound == pytest.approx(1.5)


def test_norm_preserving_primitive_keeps_the_domain_bound() -> None:
    @pjx.jit
    def clipped_negative(x: jax.Array) -> jax.Array:
        return -pj.clip_norm(x, 1.5)

    result = clipped_negative(
        _private_array(np.arange(6).reshape(2, 3))
    )

    assert isinstance(result, pjx.PrivArray)
    norm_bound = result._domain.norm_bound
    assert norm_bound is not None
    assert norm_bound.max() == pytest.approx(1.5)


def test_repeated_jit_clip_calls_have_independent_l2_constraints() -> None:
    @pjx.jit(static_argnames="bound")
    def clipped_sum(
        x: jax.Array,
        bound: float,
    ) -> jax.Array:
        return pj.clip_norm(x, bound).sum(axis=0)

    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-repeated-clip-{uuid.uuid4()}")
    private = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=accountant,
    )

    first = clipped_sum(private, bound=1.0)
    second = clipped_sum(private, bound=1.0)

    assert isinstance(first, pjx.SensitiveArray)
    assert isinstance(second, pjx.SensitiveArray)
    released = pj.gaussian_mechanism(
        (first, second),
        scale=2.0,
    )
    assert all(isinstance(value, jax.Array) for value in released)
    assert accountant.budget_spent == {2.0: pytest.approx(0.5)}


def test_joint_gaussian_combines_independent_l2_constraints() -> None:
    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-noise-groups-{uuid.uuid4()}")
    first = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=accountant,
    )
    second = _aligned_private_array(
        np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        first,
    )
    first_sum = pj.clip_norm(first, 1.0).sum(axis=0)
    second_sum = pj.clip_norm(second, 1.0).sum(axis=0)
    assert isinstance(first_sum, pjx.SensitiveArray)
    assert isinstance(second_sum, pjx.SensitiveArray)

    released = pj.gaussian_mechanism(
        {"first": first_sum, "second": second_sum},
        scale=2.0,
    )

    assert all(isinstance(leaf, jax.Array) for leaf in released.values())
    assert accountant.budget_spent == {2.0: pytest.approx(0.5)}


def test_gaussian_accounts_for_joint_and_independent_clip_constraints() -> None:
    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-noise-cache-groups-{uuid.uuid4()}")
    first = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=accountant,
    )
    second = _aligned_private_array(
        np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        first,
    )

    jointly_clipped = pj.clip_norm(
        {"first": first, "second": second},
        1.0,
    )
    joint_sums = jax.tree.map(lambda leaf: leaf.sum(axis=0), jointly_clipped)
    separate_sums = (
        pj.clip_norm(first, 1.0).sum(axis=0),
        pj.clip_norm(second, 1.0).sum(axis=0),
    )
    jointly_clipped_again = pj.clip_norm(
        {"first": first, "second": second},
        1.0,
    )
    joint_sums_again = jax.tree.map(
        lambda leaf: leaf.sum(axis=0),
        jointly_clipped_again,
    )

    @pjx.jit(static_argnames="scale")
    def release(
        first_sum: jax.Array,
        second_sum: jax.Array,
        scale: float,
    ) -> tuple[jax.Array, jax.Array]:
        return cast(
            tuple[jax.Array, jax.Array],
            pj.gaussian_mechanism(
                (first_sum, second_sum),
                scale=scale,
            ),
        )

    release(joint_sums["first"], joint_sums["second"], scale=2.0)
    release(*separate_sums, scale=2.0)
    release(
        joint_sums_again["first"],
        joint_sums_again["second"],
        scale=2.0,
    )

    assert accountant.budget_spent == {2.0: pytest.approx(1.0)}


def test_gaussian_release_supports_an_empty_private_batch() -> None:
    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-empty-noise-{uuid.uuid4()}")
    private = _private_array(
        np.empty((0, 3), dtype=np.float32),
        accountant=accountant,
    )

    @pjx.jit(static_argnames=("clip_bound", "noise_scale"))
    def release(
        x: jax.Array,
        clip_bound: float,
        noise_scale: float,
    ) -> jax.Array:
        clipped = pj.clip_norm(x, clip_bound)
        return cast(
            jax.Array,
            pj.gaussian_mechanism(
                clipped.sum(axis=0),
                scale=noise_scale,
            ),
        )

    result = release(private, clip_bound=1.0, noise_scale=2.0)

    assert isinstance(result, jax.Array)
    assert result.shape == (3,)
    assert accountant.budget_spent == {2.0: pytest.approx(0.25)}


def test_gaussian_release_honors_accountant_budget_limit() -> None:
    accountant = pj.RDPAccountant(
        alpha=[2.0],
        budget_limit={2.0: 0.1},
    )
    accountant.set_as_root(name=f"jax-noise-limit-{uuid.uuid4()}")
    private = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=accountant,
    )

    @pjx.jit
    def release(x: jax.Array) -> jax.Array:
        sensitive = pj.clip_norm(x, 1.0).sum(axis=0)
        return cast(
            jax.Array,
            pj.gaussian_mechanism(sensitive, scale=2.0),
        )

    with pytest.raises(pj.BudgetExceededError):
        release(private)

    assert accountant.budget_spent == {2.0: 0.0}


def test_jit_release_charges_each_sampled_accountant() -> None:
    root = pj.RDPAccountant(alpha=[2.0])
    root.set_as_root(name=f"jax-sampled-cache-{uuid.uuid4()}")
    first_accountant = root.create_subsampling_accountant(0.5)
    second_accountant = root.create_subsampling_accountant(0.5)
    first = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=first_accountant,
    )
    second = _private_array(
        np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        accountant=second_accountant,
    )

    @pjx.jit
    def release(x: jax.Array) -> jax.Array:
        sensitive = pj.clip_norm(x, 1.0).sum(axis=0)
        return cast(
            jax.Array,
            pj.gaussian_mechanism(sensitive, scale=2.0),
        )

    release(first)
    release(second)

    first_spent = first_accountant.budget_spent[2.0]
    second_spent = second_accountant.budget_spent[2.0]
    assert first_spent > 0
    assert second_spent == pytest.approx(first_spent)
    assert root.budget_spent[2.0] == pytest.approx(
        first_spent + second_spent
    )


def test_jit_release_respects_each_accountants_rdp_parameters() -> None:
    first_root = pj.RDPAccountant(alpha=[2.0])
    first_root.set_as_root(name=f"jax-sampled-first-{uuid.uuid4()}")
    second_root = pj.RDPAccountant(alpha=[2.0])
    second_root.set_as_root(name=f"jax-sampled-second-{uuid.uuid4()}")
    third_root = pj.RDPAccountant(alpha=[3.0])
    third_root.set_as_root(name=f"jax-sampled-third-{uuid.uuid4()}")
    first = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=first_root.create_subsampling_accountant(0.25),
    )
    second = _private_array(
        np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        accountant=second_root.create_subsampling_accountant(0.5),
    )
    third = _private_array(
        np.arange(12, 18, dtype=np.float32).reshape(2, 3),
        accountant=third_root.create_subsampling_accountant(0.5),
    )

    @pjx.jit
    def release(x: jax.Array) -> jax.Array:
        sensitive = pj.clip_norm(x, 1.0).sum(axis=0)
        return cast(
            jax.Array,
            pj.gaussian_mechanism(sensitive, scale=2.0),
        )

    release(first)
    release(second)
    release(third)

    first_spent = first_root.budget_spent[2.0]
    second_spent = second_root.budget_spent[2.0]
    third_spent = third_root.budget_spent[3.0]
    assert 0 < first_spent < second_spent
    assert third_spent > 0


def test_jit_rejects_multiple_actual_accountants() -> None:
    first_accountant = pj.RDPAccountant(alpha=[2.0])
    first_accountant.set_as_root(
        name=f"jax-first-accountant-{uuid.uuid4()}"
    )
    second_accountant = pj.RDPAccountant(alpha=[2.0])
    second_accountant.set_as_root(
        name=f"jax-second-accountant-{uuid.uuid4()}"
    )
    first = _private_array(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        accountant=first_accountant,
    )
    second = pjx.PrivArray(
        np.arange(6, 12, dtype=np.float32).reshape(2, 3),
        distance=first._distance,
        privacy_axis=0,
        parents=[first],
        accountant=second_accountant,
        keep_alignment=True,
    )

    @pjx.jit
    def add(x: jax.Array, y: jax.Array) -> jax.Array:
        return x + y

    with pytest.raises(
        pj.DPError,
        match="must share one accountant",
    ):
        add(first, second)


def test_clip_bound_argument_must_be_static() -> None:
    def clip(
        gradient: jax.Array,
        bound: float,
    ) -> jax.Array:
        return pj.clip_norm(gradient, bound)

    wrapped = pjx.jit(clip)
    private = _private_array(np.arange(6).reshape(2, 3))

    with pytest.raises(TypeError, match="bound must be static"):
        wrapped(private, 1.0)


def test_jit_rejects_an_invalid_clip_bound() -> None:
    @pjx.jit
    def invalid_clip(x: jax.Array) -> jax.Array:
        return pj.clip_norm(x, -1.0)

    private = _private_array(np.arange(6).reshape(2, 3))

    with pytest.raises(ValueError, match="bound must be finite and > 0"):
        invalid_clip(private)


def test_clip_norm_sanitizes_nonfinite_values_in_eager_and_jit() -> None:
    def clip(x: jax.Array) -> jax.Array:
        return pj.clip_norm(x, 2.0)

    private = _private_array([np.nan, np.inf, -np.inf, 4.0])
    eager = clip(cast(jax.Array, private))
    jitted = pjx.jit(clip)(private)

    assert isinstance(eager, pjx.PrivArray)
    assert isinstance(jitted, pjx.PrivArray)
    expected = [0.0, 0.0, 0.0, 2.0]
    np.testing.assert_allclose(_host(eager._value), expected)
    np.testing.assert_allclose(_host(jitted._value), expected)
    assert np.all(np.isfinite(_host(eager._value)))
    assert np.all(np.isfinite(_host(jitted._value)))


def test_array_operations_match_in_eager_and_jit_modes() -> None:
    def transform(x: jax.Array) -> jax.Array:
        reshaped = x.reshape((x.shape[0], 6))
        shifted_and_scaled = (reshaped + 1.0) * 2.0
        transposed = shifted_and_scaled.transpose((1, 0))
        return transposed.sum(axis=0)

    private = _private_array(np.arange(12).reshape(2, 2, 3))
    eager = transform(cast(jax.Array, private))
    jitted = pjx.jit(transform)(private)

    assert isinstance(eager, pjx.PrivArray)
    assert isinstance(jitted, pjx.PrivArray)
    np.testing.assert_allclose(_host(eager._value), [42.0, 114.0])
    np.testing.assert_allclose(_host(jitted._value), [42.0, 114.0])
    assert eager._alignment_signature == private._alignment_signature
    assert jitted._alignment_signature == private._alignment_signature


def test_jit_accepts_public_inputs() -> None:
    @pjx.jit
    def public_graph(x: jax.Array) -> jax.Array:
        return ((x + 1.0) * 2.0).sum(axis=1)

    result = public_graph(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))

    assert isinstance(result, jax.Array)
    np.testing.assert_allclose(_host(result), [12.0, 30.0])


def test_public_value_with_private_shape_stays_private() -> None:
    @pjx.jit
    def private_length_zeros(x: jax.Array) -> jax.Array:
        return jnp.zeros((x.shape[0],), dtype=x.dtype)

    private = _private_array(np.arange(6).reshape(2, 3))
    result = private_length_zeros(private)

    assert isinstance(result, pjx.PrivArray)
    np.testing.assert_allclose(_host(result._value), [0.0, 0.0])
    assert result._privacy_axis == 0
    assert result._alignment_signature == private._alignment_signature
    assert result._accountant is private._accountant


def test_private_shape_cannot_be_broadcast_to_a_public_axis() -> None:
    @pjx.jit
    def duplicate_private_shape(x: jax.Array) -> jax.Array:
        expanded = jnp.broadcast_to(
            x,
            (x.shape[0], x.shape[0]),
        )
        return pj.normalize(expanded).sum(axis=0)

    private = _private_array(np.arange(3).reshape(3, 1))

    with pytest.raises(
        pj.DPError,
        match="only preserve the aligned privacy dimension",
    ):
        duplicate_private_shape(private)


def test_dynamic_public_operand_can_be_mixed_with_private_input() -> None:
    @pjx.jit
    def add_bias(x: jax.Array, bias: jax.Array) -> jax.Array:
        return (x + bias).sum(axis=1)

    private = _private_array(np.arange(6).reshape(2, 3))
    bias = jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float32)
    result = add_bias(private, bias)

    assert isinstance(result, pjx.PrivArray)
    np.testing.assert_allclose(_host(result._value), [63.0, 72.0])
    assert result._domain.value_range is None


def test_public_operand_cannot_match_the_private_record_count() -> None:
    private = _private_array(np.arange(6).reshape(2, 3))

    for public_rows in (2, 3):
        public = jnp.ones((public_rows, 3), dtype=jnp.float32)
        with pytest.raises(
            pj.DPError,
            match="cannot match a public dimension against the private record count",
        ):
            private + public


def test_dp_pca_covariance_matches_in_eager_and_jit_modes() -> None:
    def covariance(x: jax.Array) -> jax.Array:
        normalized = pj.normalize(x)
        return normalized.T @ normalized

    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-pca-{uuid.uuid4()}")
    private = _private_array(
        np.asarray(
            [
                [3.0, 4.0],
                [0.0, 2.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        accountant=accountant,
    )

    eager_covariance = covariance(cast(jax.Array, private))
    jitted_covariance = pjx.jit(covariance)(private)

    assert isinstance(eager_covariance, pjx.SensitiveArray)
    assert isinstance(jitted_covariance, pjx.SensitiveArray)
    expected = np.asarray(
        [
            [1.36, 0.48],
            [0.48, 1.64],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(_host(eager_covariance._value), expected)
    np.testing.assert_allclose(_host(jitted_covariance._value), expected)
    assert eager_covariance._norm_type == "l2"
    assert jitted_covariance._norm_type == "l2"
    assert eager_covariance._distance.max() == pytest.approx(1.0)
    assert jitted_covariance._distance.max() == pytest.approx(1.0)


def test_normalize_treats_vector_elements_as_individual_records() -> None:
    def normalize(x: jax.Array) -> jax.Array:
        return pj.normalize(x)

    private = _private_array([2.0, -4.0, 0.0, np.nan, np.inf])
    eager = normalize(cast(jax.Array, private))
    jitted = pjx.jit(normalize)(private)

    assert isinstance(eager, pjx.PrivArray)
    assert isinstance(jitted, pjx.PrivArray)
    expected = [1.0, -1.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(_host(eager._value), expected)
    np.testing.assert_allclose(_host(jitted._value), expected)
    assert np.all(np.isfinite(_host(eager._value)))
    assert np.all(np.isfinite(_host(jitted._value)))


def test_dp_pca_covariance_requires_a_per_record_norm_bound() -> None:
    private = _private_array(
        np.arange(12, dtype=np.float32).reshape(4, 3)
    )

    with pytest.raises(pj.DPError, match="no per-record norm bound"):
        _ = private.T @ private

    @pjx.jit
    def unbounded_covariance(x: jax.Array) -> jax.Array:
        return x.T @ x

    with pytest.raises(pj.DPError, match="no per-record norm bound"):
        unbounded_covariance(private)


def test_dp_pca_gaussian_release_accounts_in_eager_and_jit_modes() -> None:
    def release(
        x: jax.Array,
        scale: float,
    ) -> jax.Array:
        normalized = pj.normalize(x)
        covariance = normalized.T @ normalized
        return cast(
            jax.Array,
            pj.gaussian_mechanism(covariance, scale=scale),
        )

    accountant = pj.RDPAccountant(alpha=[2.0])
    accountant.set_as_root(name=f"jax-pca-release-{uuid.uuid4()}")
    private = _private_array(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        accountant=accountant,
    )

    eager = release(cast(jax.Array, private), 2.0)
    jitted = pjx.jit(release, static_argnames="scale")(
        private,
        scale=2.0,
    )

    assert isinstance(eager, jax.Array)
    assert isinstance(jitted, jax.Array)
    assert eager.shape == (3, 3)
    assert jitted.shape == (3, 3)
    assert accountant.budget_spent == {2.0: pytest.approx(0.5)}


def test_static_argnames_specializes_privacy_affecting_scalar() -> None:
    def scaled_rows(
        x: jax.Array,
        scale: float,
    ) -> jax.Array:
        return (x * scale).sum(axis=1)

    wrapped = pjx.jit(scaled_rows, static_argnames="scale")
    private = _private_array(
        np.arange(6).reshape(2, 3),
        value_range=(0.0, 5.0),
    )

    twice = wrapped(private, scale=2.0)
    triple = wrapped(private, scale=3.0)
    twice_again = wrapped(private, scale=2.0)

    assert isinstance(twice, pjx.PrivArray)
    assert isinstance(triple, pjx.PrivArray)
    assert isinstance(twice_again, pjx.PrivArray)
    np.testing.assert_allclose(_host(twice._value), [6.0, 24.0])
    np.testing.assert_allclose(_host(triple._value), [9.0, 36.0])
    np.testing.assert_allclose(_host(twice_again._value), [6.0, 24.0])
    assert twice._domain.value_range == (0.0, 30.0)
    assert triple._domain.value_range == (0.0, 45.0)


def test_private_static_argument_is_rejected() -> None:
    private = _private_array(np.arange(6).reshape(2, 3))

    def identity(x: jax.Array) -> jax.Array:
        return x

    wrapped = pjx.jit(identity, static_argnums=0)
    with pytest.raises(TypeError, match="cannot be static"):
        wrapped(private)


def test_sensitive_array_shape_can_be_used_for_tracing() -> None:
    private = _private_array(np.arange(6).reshape(2, 3))
    normalized = pj.normalize(private)
    sensitive = normalized.sum(axis=0)
    assert isinstance(sensitive, pjx.SensitiveArray)

    @pjx.jit
    def identity(x: jax.Array) -> jax.Array:
        return x

    result = identity(sensitive)

    assert isinstance(result, pjx.SensitiveArray)
    assert result.shape == (3,)
    assert result._accountant is sensitive._accountant
    assert result._distance == sensitive._distance


def test_unsupported_jax_primitive_is_rejected() -> None:
    private = _private_array(np.arange(6).reshape(2, 3))

    @pjx.jit
    def sine(x: jax.Array) -> jax.Array:
        return jnp.sin(x)

    with pytest.raises(NotImplementedError, match="Unsupported JAX primitive: sin"):
        sine(private)


def test_call_batch_rejects_an_embedded_protected_value() -> None:
    private = _private_array(np.arange(6).reshape(2, 3))
    sensitive = pj.clip_norm(private, 1.0).sum(axis=0)
    assert isinstance(sensitive, pjx.SensitiveArray)
    release = egrpc.call(
        trusted_gaussian_mechanism,
        arrays=[sensitive],
        scale=2.0,
        delta=None,
    )
    packed = egrpc.call(
        pack_outputs,
        values=[egrpc.ValueRef(0)],
    )
    batch = egrpc.CallBatch(
        calls=[release, packed],
        output=egrpc.ValueRef(1),
    )

    with pytest.raises(
        pj.DPError,
        match="through inputs",
    ):
        validate_call_batch(batch)


def test_reduce_privacy_axis_requires_a_norm_bound() -> None:
    private = _private_array(np.arange(6).reshape(2, 3))

    @pjx.jit
    def sum_rows(x: jax.Array) -> jax.Array:
        return x.sum(axis=0)

    with pytest.raises(pj.DPError, match="Norm bound is not set"):
        sum_rows(private)


def test_eager_reshape_does_not_accept_a_concrete_private_size() -> None:
    private = _private_array(np.arange(6).reshape(2, 3))

    for private_size in (2, 3):
        with pytest.raises(
            pj.DPError,
            match="must infer its privacy dimension with -1",
        ):
            private.reshape((private_size, 3))


def test_eager_reshape_validates_record_width_independently_of_batch_size() -> None:
    for private_size in (2, 3):
        private = _private_array(
            np.arange(3 * private_size).reshape(private_size, 3)
        )
        with pytest.raises(pj.DPError, match="mix data across individuals"):
            private.reshape((-1, 2))


def test_eager_reshape_can_split_records() -> None:
    private = _private_array(np.arange(6).reshape(3, 2))

    split = private.reshape((2 * private.shape[0], 1))
    private_dimension = split.shape[0]

    assert isinstance(private_dimension, pj.SensitiveDimInt)
    assert private_dimension._value == 6
    assert split._distance.max() == pytest.approx(2.0)
    assert (
        split._alignment_signature.base
        == private._alignment_signature.base
    )
    assert split._alignment_signature.left == 1
    assert split._alignment_signature.right == 2


def test_eager_reshape_tracks_a_left_record_factor() -> None:
    private = _private_array(
        np.arange(6).reshape(2, 3),
        privacy_axis=1,
    )

    split = private.reshape((2 * private.shape[1], 1))
    private_dimension = split.shape[0]

    assert isinstance(private_dimension, pj.SensitiveDimInt)
    assert private_dimension._value == 6
    assert split._distance.max() == pytest.approx(2.0)
    assert (
        split._alignment_signature.base
        == private._alignment_signature.base
    )
    assert split._alignment_signature.left == 2
    assert split._alignment_signature.right == 1


def test_reshape_that_moves_the_private_dimension_is_rejected() -> None:
    private = _private_array(np.arange(24).reshape(4, 2, 3))

    @pjx.jit
    def unsafe_reshape(x: jax.Array) -> jax.Array:
        return x.reshape((2, x.shape[0], 3))

    with pytest.raises(pj.DPError, match="mix data across individuals"):
        unsafe_reshape(private)


def test_reshape_split_derives_alignment_and_scales_distance() -> None:
    @pjx.jit
    def split_records(x: jax.Array) -> jax.Array:
        normalized = pj.normalize(x)
        return normalized.reshape((2 * x.shape[0], 1))

    private = _private_array(np.arange(6).reshape(3, 2))
    split = split_records(private)
    second = split_records(private)

    assert isinstance(split, pjx.PrivArray)
    assert split.shape[0]._value == 6
    assert split._distance.max() == pytest.approx(2.0)
    assert split._domain.norm_bound is not None
    assert split._domain.norm_bound.max() == pytest.approx(1.0)
    assert split._alignment_signature != private._alignment_signature
    assert (
        split._alignment_signature.base
        == private._alignment_signature.base
    )
    assert split._alignment_signature.left == 1
    assert split._alignment_signature.right == 2
    assert second._alignment_signature == split._alignment_signature


def test_reshape_can_restore_an_exact_old_alignment() -> None:
    @pjx.jit
    def split_and_pack(x: jax.Array) -> jax.Array:
        split = x.reshape((2 * x.shape[0], 2))
        return split.reshape((x.shape[0], 4))

    private = _private_array(np.arange(12).reshape(3, 4))

    result = split_and_pack(private)

    assert isinstance(result, pjx.PrivArray)
    np.testing.assert_array_equal(
        _host(result._value),
        _host(private._value),
    )
    assert result._alignment_signature == private._alignment_signature
    assert result._distance.structurally_equal(private._distance)
    assert result._domain.norm_bound is None


def test_reshape_cannot_move_a_factor_across_the_record_base() -> None:
    @pjx.jit
    def move_right_factor_to_left(x: jax.Array) -> jax.Array:
        split = x.reshape((2 * x.shape[0], 1))
        return split.reshape((2, x.shape[0], 1))

    private = _private_array(np.arange(6).reshape(3, 2))

    with pytest.raises(
        pj.DPError,
        match="mix data across individuals",
    ):
        move_right_factor_to_left(private)


def test_jit_reshape_binds_the_derived_base_to_each_input() -> None:
    @pjx.jit
    def split_records(x: jax.Array) -> jax.Array:
        return x.reshape((2 * x.shape[0], 1))

    first = _private_array(np.arange(6).reshape(3, 2))
    second = _private_array(np.arange(8).reshape(4, 2))

    first_result = split_records(first)
    second_result = split_records(second)

    assert isinstance(first_result, pjx.PrivArray)
    assert isinstance(second_result, pjx.PrivArray)
    assert (
        first_result._alignment_signature.base
        == first._alignment_signature.base
    )
    assert (
        second_result._alignment_signature.base
        == second._alignment_signature.base
    )
    assert (
        first_result._alignment_signature.base
        != second_result._alignment_signature.base
    )
    assert first_result._alignment_signature.right == 2
    assert second_result._alignment_signature.right == 2


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 3), [6.0, 24.0]),
        ((5, 3), [6.0, 24.0, 42.0, 60.0, 78.0]),
    ],
)
def test_jit_accepts_concrete_private_batch_sizes(
    shape: tuple[int, int],
    expected: Sequence[float],
) -> None:
    @pjx.jit
    def row_sum_twice(x: jax.Array) -> jax.Array:
        return (x * 2.0).sum(axis=1)

    private = _private_array(np.arange(np.prod(shape)).reshape(shape))
    result = row_sum_twice(private)

    assert isinstance(result, pjx.PrivArray)
    np.testing.assert_allclose(_host(result._value), expected)
