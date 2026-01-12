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

import math
import uuid

import pytest

import numpy as _np

import privjail as pj
import privjail.numpy as pnp

@pytest.fixture
def accountant() -> pj.ApproxAccountant:
    acc = pj.ApproxAccountant()
    acc.set_as_root(name=str(uuid.uuid4()))
    return acc

def test_gaussian_mechanism_over_rows(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0, -1.0, 0.5],
                           [2.0, -2.0, 1.0],
                           [3.0, -3.0, 1.5],
                           [0.0,  2.0, 1.2]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    assert arr.ndim == 2
    assert isinstance(arr.shape[0], pj.SensitiveInt)
    assert all(isinstance(x, int) for x in arr.shape[1:])
    assert arr.shape[0]._value == 4
    assert arr.shape[1] == 3

    bound = 5.0
    clipped = pj.clip_norm(arr, bound=bound, ord=2)

    assert isinstance(clipped, pnp.PrivNDArray)
    assert clipped.domain.norm_type == "l2"
    assert clipped.domain.norm_bound == pytest.approx(bound)
    assert clipped.ndim == 2
    assert isinstance(clipped.shape[0], pj.SensitiveInt)
    assert all(isinstance(x, int) for x in clipped.shape[1:])
    assert clipped.shape[0]._value == 4
    assert clipped.shape[1] == 3

    clipped_sum = clipped.sum(axis=0)

    assert isinstance(clipped_sum, pnp.SensitiveNDArray)
    assert clipped_sum.shape == (3,)
    assert clipped_sum.norm_type == "l2"
    assert clipped_sum.max_distance == pytest.approx(bound)

    noisy_sum = pj.gaussian_mechanism(clipped_sum, eps=1.0, delta=1e-5)

    assert isinstance(noisy_sum, _np.ndarray)
    assert noisy_sum.shape == (3,)

    spent_kind, spent_budget = pj.budgets_spent()[accountant._root_name]
    assert spent_kind == "approx"
    assert spent_budget == pytest.approx((1.0, 1e-5))

def test_transpose(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    arr_t = arr.transpose((1, 0))

    assert arr_t.distance_axis == 1
    assert arr_t.axis_signature == arr.axis_signature
    assert arr_t._value.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    arr_tt = arr_t.T

    assert arr_tt.distance_axis == 0
    assert arr_tt.axis_signature == arr.axis_signature
    assert arr_tt._value.tolist() == [[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]]

def test_swapaxes(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[[1.0, 2.0], [3.0, 4.0]],
                           [[5.0, 6.0], [7.0, 8.0]]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    swapped_02 = arr.swapaxes(0, 2)

    assert swapped_02.distance_axis == 2
    assert swapped_02.axis_signature == arr.axis_signature
    assert swapped_02._value.tolist() == [[[1.0, 5.0], [3.0, 7.0]],
                                          [[2.0, 6.0], [4.0, 8.0]]]

    swapped_11 = arr.swapaxes(1, 1)
    assert swapped_11.distance_axis == 0
    assert swapped_11.axis_signature == arr.axis_signature
    assert swapped_11._value.tolist() == arr._value.tolist()

    swapped_12 = arr.swapaxes(1, 2)
    assert swapped_12.distance_axis == 0
    assert swapped_12.axis_signature == arr.axis_signature
    assert swapped_12._value.tolist() == [[[1.0, 3.0], [2.0, 4.0]],
                                          [[5.0, 7.0], [6.0, 8.0]]]

def test_clip_norm_scalar_rows(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([1.0, -2.5, 0.5, 4.0, -6.0],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    bound = 2.0
    clipped = pj.clip_norm(arr, bound=bound, ord=2)

    assert clipped._value.tolist() == pytest.approx([1.0, -2.0, 0.5, 2.0, -2.0])
    assert arr._value.tolist() == [1.0, -2.5, 0.5, 4.0, -6.0]

def test_clip_norm_matrix_rows(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[ 3.0, 4.0,  0.0],
                           [ 0.0, 0.0,  0.0],
                           [ 6.0, 8.0,  2.0],
                           [-1.0, 2.0, -2.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    bound = 5.0
    clipped = pj.clip_norm(arr, bound=bound, ord=2)

    clipped_rows = clipped._value.tolist()
    norms = [math.sqrt(sum(v * v for v in row)) for row in clipped_rows]

    assert norms[0] == pytest.approx(bound)
    assert norms[1] == pytest.approx(0.0)
    assert norms[2] == pytest.approx(bound)
    assert norms[3] == pytest.approx(3.0)
    assert arr._value.tolist() == [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [6.0, 8.0, 2.0], [-1.0, 2.0, -2.0]]

def test_clip_norm_tensor_rows(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[[ 3.0, 0.0, -4.0], [0.0,  0.0,  0.0]],
                           [[-5.0, 0.0,  0.0], [0.0, 12.0,  0.0]],
                           [[ 0.0, 0.0,  0.0], [8.0,  0.0, 15.0]]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    bound = 6.0
    clipped = pj.clip_norm(arr, bound=bound, ord=2)

    assert arr._value.shape == clipped._value.shape

    original_rows = arr._value.reshape(arr._value.shape[0], -1)
    clipped_rows = clipped._value.reshape(clipped._value.shape[0], -1)

    original_norms = _np.linalg.norm(original_rows, ord=2, axis=1).tolist()
    clipped_norms = _np.linalg.norm(clipped_rows, ord=2, axis=1).tolist()

    assert original_norms[0] == pytest.approx(5.0)
    assert original_norms[1] == pytest.approx(13.0)
    assert original_norms[2] == pytest.approx(17.0)

    assert clipped_norms[0] == pytest.approx(original_norms[0])
    assert clipped_norms[1] == pytest.approx(bound)
    assert clipped_norms[2] == pytest.approx(bound)

    expected = [[[ 3.0, 0.0, -4.0], [0.0,  0.0,  0.0]],
                [[-5.0, 0.0,  0.0], [0.0, 12.0,  0.0]],
                [[ 0.0, 0.0,  0.0], [8.0,  0.0, 15.0]]]
    assert arr._value.tolist() == expected

# def test_clip_norm_axis_argument(accountant: pj.ApproxAccountant) -> None:
#     arr = pnp.PrivNDArray([[1.0, 2.0], [3.0, 4.0]],
#                           distance      = pj.RealExpr(1),
#                           distance_axis = 0,
#                           accountant    = accountant)

#     clipped = arr.clip_norm(bound=2.5, ord=2, axis=0)
#     clipped_t = arr.T.clip_norm(bound=2.5, ord=2, axis=1)

#     assert clipped.domain.norm_type == "l2"
#     assert clipped.domain.norm_bound == 2.5
#     assert clipped_t.domain.norm_type == "l2"
#     assert clipped_t.domain.norm_bound == 2.5

#     assert _np.allclose(clipped._value, clipped_t.T._value)

#     with pytest.raises(pj.DPError):
#         arr.clip_norm(bound=2.5, ord=2, axis=1)

def test_normalize(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[3.0, 4.0],
                           [0.0, 5.0],
                           [1.0, 0.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)
    normalized = pj.normalize(arr, ord=2)
    expected = [[3.0/5.0, 4.0/5.0],
                [0.0, 1.0],
                [1.0, 0.0]]
    assert _np.allclose(normalized._value, expected)
    assert normalized._domain.norm_type == "l2"
    assert normalized._domain.norm_bound == 1.0
    assert normalized._domain.value_range == (-1.0, 1.0)
    assert normalized.distance == arr.distance
    assert normalized.axis_signature == arr.axis_signature
    row_norms = _np.linalg.norm(normalized._value, ord=2, axis=1)
    assert _np.allclose(row_norms, [1.0, 1.0, 1.0])

def test_sample(accountant: pj.ApproxAccountant) -> None:
    x = pnp.PrivNDArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                        distance      = pj.RealExpr(1),
                        distance_axis = 0,
                        accountant    = accountant)

    # Single array
    (x_s,) = pj.sample(x, q=0.5)
    assert isinstance(x_s, pnp.PrivNDArray)
    assert x_s.axis_signature != x.axis_signature
    assert x_s._value.shape[0] <= x._value.shape[0]

    # Multiple arrays
    y = x * 2  # same axis_signature
    x_s, y_s = pj.sample(x, y, q=0.5)
    assert isinstance(x_s, pnp.PrivNDArray)
    assert isinstance(y_s, pnp.PrivNDArray)
    assert x_s.axis_signature == y_s.axis_signature
    assert x_s.axis_signature != x.axis_signature
    assert _np.allclose(y_s._value, x_s._value * 2)
    assert x_s._value.shape[0] <= x._value.shape[0]
    assert y_s._value.shape[0] == x_s._value.shape[0]

def test_normalize_tensor(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[[ 3.0, 0.0, -4.0], [0.0,  0.0,  0.0]],
                           [[-5.0, 0.0,  0.0], [0.0, 12.0,  0.0]],
                           [[ 0.0, 0.0,  0.0], [8.0,  0.0, 15.0]]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)
    normalized = pj.normalize(arr, ord=2)
    assert arr._value.shape == normalized._value.shape
    original_rows = arr._value.reshape(arr._value.shape[0], -1)
    normalized_rows = normalized._value.reshape(normalized._value.shape[0], -1)
    original_norms = _np.linalg.norm(original_rows, ord=2, axis=1)
    normalized_norms = _np.linalg.norm(normalized_rows, ord=2, axis=1)
    assert original_norms.tolist() == pytest.approx([5.0, 13.0, 17.0])
    assert _np.allclose(normalized_norms, [1.0, 1.0, 1.0])
    assert normalized._domain.norm_type == "l2"
    assert normalized._domain.norm_bound == 1.0
    assert normalized._domain.value_range == (-1.0, 1.0)

def test_sum_returns_sensitive_float(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([1.0, -1.5, 0.5, 2.0],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    bound = 2.0
    clipped = pj.clip_norm(arr, bound=bound, ord=1)
    total = clipped.sum(axis=0)

    assert isinstance(total, pj.SensitiveFloat)
    assert total._value == pytest.approx(clipped._value.sum())
    assert total.max_distance == pytest.approx(bound)

def test_sum_returns_sensitive_ndarray(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0,  2.0, -1.5],
                           [0.5, -4.0,  3.0],
                           [2.0,  1.0, -0.5]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    bound = 3.0
    clipped = pj.clip_norm(arr, bound=bound, ord=1)
    summed = clipped.sum(axis=0)

    assert isinstance(summed, pnp.SensitiveNDArray)
    assert summed.norm_type == "l1"
    assert summed.shape == (3,)

    expected = _np.asarray(clipped._value.sum(axis=0), dtype=float)
    assert summed._value.tolist() == pytest.approx(expected.tolist())
    assert summed.max_distance == pytest.approx(bound)

def test_max_min(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0, 5.0, 3.0],
                           [9.0, 2.0, 4.0],
                           [6.0, 8.0, 7.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          domain        = pnp.NDArrayDomain(value_range=(0.0, 10.0)),
                          accountant    = accountant)

    for method, expected, expected_kd in [
        (arr.max, [5.0, 9.0, 8.0], [[5.0], [9.0], [8.0]]),
        (arr.min, [1.0, 2.0, 6.0], [[1.0], [2.0], [6.0]]),
    ]:
        # along axis=1 (non-distance axis)
        result = method(axis=1)
        assert isinstance(result, pnp.PrivNDArray)
        assert _np.allclose(result._value, expected)
        assert result._value.shape == (3,)
        assert result.distance_axis == 0
        assert result.domain.value_range == (0.0, 10.0)
        assert result.axis_signature == arr.axis_signature

        # with keepdims=True
        result_kd = method(axis=1, keepdims=True)
        assert _np.allclose(result_kd._value, expected_kd)
        assert result_kd._value.shape == (3, 1)
        assert result_kd.distance_axis == 0
        assert result_kd.axis_signature == arr.axis_signature

        # along distance axis should raise DPError
        with pytest.raises(pj.DPError):
            method(axis=0)

        # axis=None should raise DPError
        with pytest.raises(pj.DPError):
            method(axis=None)

        # negative axis
        result_neg = method(axis=-1)
        assert _np.allclose(result_neg._value, expected)

    # 3D array: distance_axis adjustment
    arr3d = pnp.PrivNDArray([[[1.0, 2.0], [3.0, 4.0]],
                             [[5.0, 6.0], [7.0, 8.0]]],
                            distance      = pj.RealExpr(1),
                            distance_axis = 2,
                            accountant    = accountant)
    # along axis=1, distance_axis=2 -> new distance_axis=1
    for method, expected3d, expected3d_kd in [
        (arr3d.max, [[3.0, 4.0], [7.0, 8.0]], [[[3.0, 4.0]], [[7.0, 8.0]]]),
        (arr3d.min, [[1.0, 2.0], [5.0, 6.0]], [[[1.0, 2.0]], [[5.0, 6.0]]]),
    ]:
        result3d = method(axis=1)
        assert result3d._value.shape == (2, 2)
        assert result3d.distance_axis == 1
        assert _np.allclose(result3d._value, expected3d)

        result3d_kd = method(axis=1, keepdims=True)
        assert result3d_kd._value.shape == (2, 1, 2)
        assert result3d_kd.distance_axis == 2
        assert _np.allclose(result3d_kd._value, expected3d_kd)

def test_linalg_norm(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([10.0, -20.0, 30.0, 40.0],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)

    clipped = pj.clip_norm(arr, bound=5.0, ord=1)
    l1norm = pnp.linalg.norm(clipped, ord=1)

    assert isinstance(l1norm, pj.SensitiveFloat)
    assert l1norm._value == pytest.approx(20.0)
    assert l1norm.max_distance == pytest.approx(5.0)

def test_matmul_priv_left(accountant: pj.ApproxAccountant) -> None:
    priv = pnp.PrivNDArray([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]],
                           distance      = pj.RealExpr(1),
                           distance_axis = 0,
                           accountant    = accountant)

    other = _np.array([[1.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 0.0]])

    result = priv @ other

    assert isinstance(result, pnp.PrivNDArray)
    assert isinstance(result.shape[0], pj.SensitiveInt)
    assert result.shape[0]._value == 2
    assert result.shape[1] == 4
    assert result.distance_axis == priv.distance_axis
    assert result.axis_signature == priv.axis_signature
    assert _np.allclose(result._value, priv._value @ other)

    with pytest.raises(pj.DPError):
        priv.T @ other

def test_matmul_priv_right(accountant: pj.ApproxAccountant) -> None:
    priv = pnp.PrivNDArray([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]],
                           distance      = pj.RealExpr(1),
                           distance_axis = 1,
                           accountant    = accountant)

    other = _np.array([[1.0, 0.0],
                       [0.0, 1.0],
                       [1.0, 1.0],
                       [2.0, 0.5]])

    result = other @ priv

    assert isinstance(result, pnp.PrivNDArray)
    assert result.shape[0] == 4
    assert isinstance(result.shape[1], pj.SensitiveInt)
    assert result.shape[1]._value == 3
    assert result.distance_axis == priv.distance_axis
    assert result.axis_signature == priv.axis_signature
    assert _np.allclose(result._value, other @ priv._value)

    with pytest.raises(pj.DPError):
        other @ priv.T

def test_matmul_priv_priv(accountant: pj.ApproxAccountant) -> None:
    # x.T @ x where x is normalized: left has distance_axis=1, right has distance_axis=0
    x = pnp.PrivNDArray([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0],
                         [10.0, 11.0, 12.0]],
                        distance      = pj.RealExpr(1),
                        distance_axis = 0,
                        accountant    = accountant)
    x_norm = pj.normalize(x, ord=2)  # (4, 3), distance_axis=0, norm_bound=1.0
    result = x_norm.T @ x_norm  # (3, 4) @ (4, 3) -> (3, 3)
    assert isinstance(result, pnp.SensitiveNDArray)
    assert result.shape == (3, 3)
    assert result.norm_type == "l2"
    assert result.max_distance == pytest.approx(1.0)
    assert _np.allclose(result._value, x_norm._value.T @ x_norm._value)

def test_sensitive_ndarray_arithmetic(accountant: pj.ApproxAccountant) -> None:
    arr1 = pnp.SensitiveNDArray([4.0, 6.0],
                                distance   = pj.RealExpr(1.0),
                                norm_type  = "l1",
                                accountant = accountant)

    arr2 = pnp.SensitiveNDArray([0.5, -1.0],
                                distance   = pj.RealExpr(5.0),
                                norm_type  = "l1",
                                accountant = accountant)

    combined = arr1 + arr2
    assert combined._value.tolist() == pytest.approx([4.5, 5.0])
    assert combined.max_distance == pytest.approx(arr1.max_distance + arr2.max_distance)

    scalar_rhs_add = arr1 + 2.0
    assert isinstance(scalar_rhs_add, pnp.SensitiveNDArray)
    assert scalar_rhs_add._value.tolist() == pytest.approx([6.0, 8.0])
    assert scalar_rhs_add.max_distance == pytest.approx(arr1.max_distance)

    scalar_lhs_add = 2.0 + arr1
    assert isinstance(scalar_lhs_add, pnp.SensitiveNDArray)
    assert scalar_lhs_add._value.tolist() == pytest.approx([6.0, 8.0])
    assert scalar_lhs_add.max_distance == pytest.approx(arr1.max_distance)

    array_rhs_add = arr1 + _np.array([1.0, 1.0])
    assert isinstance(array_rhs_add, pnp.SensitiveNDArray)
    assert array_rhs_add._value.tolist() == pytest.approx([5.0, 7.0])
    assert array_rhs_add.max_distance == pytest.approx(arr1.max_distance)

    array_lhs_add = _np.array([1.0, 1.5]) + arr1
    assert isinstance(array_lhs_add, pnp.SensitiveNDArray)
    assert array_lhs_add._value.tolist() == pytest.approx([5.0, 7.5])
    assert array_lhs_add.max_distance == pytest.approx(arr1.max_distance)

    scalar_rhs_sub = arr1 - 2.0
    assert isinstance(scalar_rhs_sub, pnp.SensitiveNDArray)
    assert scalar_rhs_sub._value.tolist() == pytest.approx([2.0, 4.0])
    assert scalar_rhs_sub.max_distance == pytest.approx(arr1.max_distance)

    scalar_lhs_sub = 2.0 - arr1
    assert isinstance(scalar_lhs_sub, pnp.SensitiveNDArray)
    assert scalar_lhs_sub._value.tolist() == pytest.approx([-2.0, -4.0])
    assert scalar_lhs_sub.max_distance == pytest.approx(arr1.max_distance)

    array_rhs_sub = arr1 - _np.array([1.0, 1.5])
    assert isinstance(array_rhs_sub, pnp.SensitiveNDArray)
    assert array_rhs_sub._value.tolist() == pytest.approx([3.0, 4.5])
    assert array_rhs_sub.max_distance == pytest.approx(arr1.max_distance)

    array_lhs_sub = _np.array([1.0, 1.5]) - arr1
    assert isinstance(array_lhs_sub, pnp.SensitiveNDArray)
    assert array_lhs_sub._value.tolist() == pytest.approx([-3.0, -4.5])
    assert array_lhs_sub.max_distance == pytest.approx(arr1.max_distance)

    unary_neg = -arr1
    assert isinstance(unary_neg, pnp.SensitiveNDArray)
    assert unary_neg._value.tolist() == pytest.approx([-4.0, -6.0])
    assert unary_neg.max_distance == pytest.approx(arr1.max_distance)

    # shape mismatch
    with pytest.raises(ValueError):
        arr1 + _np.array([1.0, 1.0, 1.0])

    # norm type mismatch
    with pytest.raises(ValueError):
        arr1 + pnp.SensitiveNDArray([0.1, 0.2],
                                    distance   = pj.RealExpr(0.5),
                                    norm_type  = "l2",
                                    accountant = accountant)

def test_maximum_minimum(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[-1.0, 2.0], [3.0, -4.0], [0.5, 0.5]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          domain        = pnp.NDArrayDomain(value_range=(-5.0, 5.0)),
                          accountant    = accountant)

    # maximum with scalar
    result = pnp.maximum(arr, 0.0)
    assert isinstance(result, pnp.PrivNDArray)
    assert _np.allclose(result._value, [[0.0, 2.0], [3.0, 0.0], [0.5, 0.5]])
    assert result.domain.value_range == (0.0, 5.0)
    assert result.axis_signature == arr.axis_signature

    # minimum with scalar
    result = pnp.minimum(arr, 1.0)
    assert isinstance(result, pnp.PrivNDArray)
    assert _np.allclose(result._value, [[-1.0, 1.0], [1.0, -4.0], [0.5, 0.5]])
    assert result.domain.value_range == (-5.0, 1.0)
    assert result.axis_signature == arr.axis_signature

    # clip pattern: maximum then minimum
    clipped = pnp.minimum(pnp.maximum(arr, -1.0), 1.0)
    assert _np.allclose(clipped._value, [[-1.0, 1.0], [1.0, -1.0], [0.5, 0.5]])
    assert clipped.domain.value_range == (-1.0, 1.0)

    # maximum/minimum between two PrivNDArrays
    arr2 = arr * 0.5  # same axis_signature
    result = pnp.maximum(arr, arr2)
    expected = _np.maximum(arr._value, arr2._value)
    assert _np.allclose(result._value, expected)
    assert result.axis_signature == arr.axis_signature

    result = pnp.minimum(arr, arr2)
    expected = _np.minimum(arr._value, arr2._value)
    assert _np.allclose(result._value, expected)

    # axis_signature mismatch should raise DPError
    other = pnp.PrivNDArray([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                            distance      = pj.RealExpr(1),
                            distance_axis = 0,
                            accountant    = accountant)
    with pytest.raises(pj.DPError):
        pnp.maximum(arr, other)
    with pytest.raises(pj.DPError):
        pnp.minimum(arr, other)

def test_exp(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[0.0, 1.0], [2.0, -1.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          domain        = pnp.NDArrayDomain(value_range=(-1.0, 2.0)),
                          accountant    = accountant)

    result = pnp.exp(arr)
    assert isinstance(result, pnp.PrivNDArray)
    assert _np.allclose(result._value, _np.exp(arr._value))
    assert result.domain.value_range is not None
    assert result.domain.value_range[0] == pytest.approx(_np.exp(-1.0))
    assert result.domain.value_range[1] == pytest.approx(_np.exp(2.0))
    assert result.axis_signature == arr.axis_signature

def test_log(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0, 2.0], [3.0, 4.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          domain        = pnp.NDArrayDomain(value_range=(0.5, 5.0)),
                          accountant    = accountant)

    result = pnp.log(arr)
    assert isinstance(result, pnp.PrivNDArray)
    assert _np.allclose(result._value, _np.log(arr._value))
    assert result.domain.value_range is not None
    assert result.domain.value_range[0] == pytest.approx(_np.log(0.5))
    assert result.domain.value_range[1] == pytest.approx(_np.log(5.0))
    assert result.axis_signature == arr.axis_signature

    # value_range with lo <= 0 becomes None
    arr_neg = pnp.PrivNDArray([[1.0, 2.0]],
                              distance      = pj.RealExpr(1),
                              distance_axis = 0,
                              domain        = pnp.NDArrayDomain(value_range=(-1.0, 5.0)),
                              accountant    = accountant)
    result_neg = pnp.log(arr_neg)
    assert result_neg.domain.value_range is None

def test_histogram_basic(accountant: pj.ApproxAccountant) -> None:
    samples = pnp.PrivNDArray([0.1, 0.4, 0.8],
                              distance      = pj.RealExpr(1.0),
                              distance_axis = 0,
                              accountant    = accountant)

    hist, edges = pnp.histogram(samples, bins=3, range=(0.0, 0.9))
    expected_hist, expected_edges = _np.histogram([0.1, 0.4, 0.8], bins=3, range=(0.0, 0.9))

    assert isinstance(hist, pnp.SensitiveNDArray)
    assert hist.norm_type == "l1"
    assert hist.max_distance == pytest.approx(samples.max_distance)
    assert _np.allclose(hist._value, expected_hist)
    assert _np.allclose(edges, expected_edges)

    explicit_edges = [0.0, 0.3, 0.6, 0.9]
    hist2, edges2 = pnp.histogram(samples, bins=explicit_edges)
    expected_hist2, expected_edges2 = _np.histogram([0.1, 0.4, 0.8], bins=explicit_edges)
    assert _np.allclose(hist2._value, expected_hist2)
    assert _np.allclose(edges2, expected_edges2)

    with pytest.raises(pj.DPError):
        pnp.histogram(samples, bins=4)

    with pytest.raises(ValueError):
        pnp.histogram(samples, bins=explicit_edges, range=(0.0, 0.9))

def test_histogramdd_basic(accountant: pj.ApproxAccountant) -> None:
    samples = pnp.PrivNDArray([[0.1, 0.2 ],
                               [0.4, 0.8 ],
                               [0.9, 0.05]],
                              distance      = pj.RealExpr(1.0),
                              distance_axis = 0,
                              accountant    = accountant)

    hist, edges = pnp.histogramdd(samples,
                                  bins  = [2, 2],
                                  range = [(0.0, 1.0), (0.0, 1.0)])

    assert isinstance(hist, pnp.SensitiveNDArray)
    assert hist.norm_type == "l1"
    assert hist.max_distance == pytest.approx(samples.max_distance)
    assert hist._value.tolist() == [[1.0, 1.0], [1.0, 0.0]]

    assert len(edges) == 2
    for edge in edges:
        assert edge.dtype == float
        assert _np.allclose(edge, _np.linspace(0.0, 1.0, num=3))

    with pytest.raises(pj.DPError):
        pnp.histogramdd(samples, bins=[2, 2])

    grid = [_np.linspace(0.0, 1.0, num=3).tolist(), _np.linspace(0.0, 1.0, num=3).tolist()]
    hist2, edges2 = pnp.histogramdd(samples, bins=grid)
    assert _np.allclose(hist2._value, hist._value)
    for e1, e2 in zip(edges, edges2, strict=True):
        assert _np.allclose(e1, e2)

    with pytest.raises(ValueError):
        pnp.histogramdd(samples,
                        bins  = grid,
                        range = [(0.0, 1.0), (0.0, 1.0)])

def test_sensitive_dim_int_basic(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)
    arr2 = pnp.PrivNDArray([[1.0, 2.0]],
                           distance      = pj.RealExpr(1),
                           distance_axis = 0,
                           accountant    = accountant)

    shape = arr.shape
    assert isinstance(shape[0], pj.SensitiveDimInt)
    assert isinstance(shape[1], int)
    assert shape[0]._value == 3
    assert shape[0].scale == 1
    assert shape[0].axis_signature == arr.axis_signature
    assert shape[1] == 3
    n = shape[0]

    # __neg__
    neg_n = -n
    assert isinstance(neg_n, pj.SensitiveDimInt)
    assert neg_n._value == -3
    assert neg_n.scale == -1
    assert neg_n.axis_signature == n.axis_signature

    # __mul__ / __rmul__
    n2 = n * 2
    assert isinstance(n2, pj.SensitiveDimInt)
    assert n2._value == 6
    assert n2.scale == 2
    n2r = 2 * n
    assert isinstance(n2r, pj.SensitiveDimInt)
    assert n2r._value == 6
    assert n2r.scale == 2
    neg_mul = n * (-2)
    assert isinstance(neg_mul, pj.SensitiveDimInt)
    assert neg_mul._value == -6
    assert neg_mul.scale == -2

    # __add__ (same signature)
    add_same = n + n2
    assert isinstance(add_same, pj.SensitiveDimInt)
    assert add_same._value == 3 + 6
    assert add_same.scale == 1 + 2
    assert add_same.axis_signature == n.axis_signature

    # __add__ (different signature)
    m = arr2.shape[0]
    assert isinstance(m, pj.SensitiveDimInt)
    assert n.axis_signature != m.axis_signature
    add_diff = n + m
    assert isinstance(add_diff, pj.SensitiveInt)
    assert not isinstance(add_diff, pj.SensitiveDimInt)
    assert add_diff._value == 3 + 1

    # __sub__ (same signature)
    sub_same = n2 - n
    assert isinstance(sub_same, pj.SensitiveDimInt)
    assert sub_same._value == 6 - 3
    assert sub_same.scale == 2 - 1

    # __sub__ (different signature)
    sub_diff = n - m
    assert isinstance(sub_diff, pj.SensitiveInt)
    assert not isinstance(sub_diff, pj.SensitiveDimInt)
    assert sub_diff._value == 3 - 1

    # __add__ with int (fallback to SensitiveInt)
    add_int = n + 10
    assert isinstance(add_int, pj.SensitiveInt)
    assert not isinstance(add_int, pj.SensitiveDimInt)
    assert add_int._value == 3 + 10

    # __sub__ with int (fallback to SensitiveInt)
    sub_int = n - 1
    assert isinstance(sub_int, pj.SensitiveInt)
    assert not isinstance(sub_int, pj.SensitiveDimInt)
    assert sub_int._value == 3 - 1

    # __add__ with float (fallback to SensitiveFloat)
    add_float = n + 0.5
    assert isinstance(add_float, pj.SensitiveFloat)
    assert add_float._value == 3 + 0.5

    # __mul__ with float (fallback to SensitiveFloat)
    mul_float = n * 1.5
    assert isinstance(mul_float, pj.SensitiveFloat)
    assert mul_float._value == 3 * 1.5

    # __add__ with SensitiveInt (fallback)
    si = pj.SensitiveInt(10, distance=pj.RealExpr(1), accountant=accountant)
    add_si = n + si
    assert isinstance(add_si, pj.SensitiveInt)
    assert not isinstance(add_si, pj.SensitiveDimInt)
    assert add_si._value == 3 + 10

    # __sub__ with SensitiveInt (fallback)
    sub_si = n - si
    assert isinstance(sub_si, pj.SensitiveInt)
    assert not isinstance(sub_si, pj.SensitiveDimInt)
    assert sub_si._value == 3 - 10

    # __add__ with SensitiveFloat (fallback)
    sf = pj.SensitiveFloat(0.5, distance=pj.RealExpr(1), accountant=accountant)
    add_sf = n + sf
    assert isinstance(add_sf, pj.SensitiveFloat)
    assert add_sf._value == 3 + 0.5

    # __sub__ with SensitiveFloat (fallback)
    sub_sf = n - sf
    assert isinstance(sub_sf, pj.SensitiveFloat)
    assert sub_sf._value == 3 - 0.5

    # reveal()
    revealed = n.reveal(eps=1.0)
    assert isinstance(revealed, (int, float))

def test_reshape_basic(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)
    n = arr.shape[0]
    assert isinstance(n, pj.SensitiveDimInt)

    # shape specified with SensitiveDimInt (scale=1)
    reshaped = arr.reshape((n, 3))
    assert reshaped._value.shape == (2, 3)
    assert reshaped.distance_axis == 0
    assert reshaped.axis_signature == arr.axis_signature

    # flatten with scaled SensitiveDimInt (n*3)
    flattened_scaled = arr.reshape((n * 3,))
    assert flattened_scaled._value.shape == (6,)
    assert flattened_scaled.distance_axis == 0
    assert flattened_scaled.max_distance == pytest.approx(arr.max_distance * 3)

    # -1 inferred as SensitiveDimInt
    reshaped_inferred = arr.reshape((-1, 3))
    assert reshaped_inferred._value.shape == (2, 3)
    assert reshaped_inferred.distance_axis == 0

    # flatten with -1
    flattened_inferred = arr.reshape((-1,))
    assert flattened_inferred._value.shape == (6,)
    assert flattened_inferred.distance_axis == 0
    assert flattened_inferred.max_distance == pytest.approx(arr.max_distance * 3)

    # SensitiveDimInt given, -1 inferred as int
    arr_3d = pnp.PrivNDArray([[[1.0, 2.0], [3.0, 4.0]],
                              [[5.0, 6.0], [7.0, 8.0]],
                              [[9.0, 10.0], [11.0, 12.0]]],
                             distance      = pj.RealExpr(1),
                             distance_axis = 0,
                             accountant    = accountant)
    k = arr_3d.shape[0]
    assert isinstance(k, pj.SensitiveDimInt)
    reshaped_3d = arr_3d.reshape((k, -1))
    assert reshaped_3d._value.shape == (3, 4)
    assert reshaped_3d.distance_axis == 0
    assert reshaped_3d.axis_signature == arr_3d.axis_signature

    # split rows (scale=2)
    arr2 = pnp.PrivNDArray([[1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0]],
                           distance      = pj.RealExpr(1),
                           distance_axis = 0,
                           accountant    = accountant)
    m = arr2.shape[0]
    assert isinstance(m, pj.SensitiveDimInt)
    reshaped_split = arr2.reshape((m * 2, 2))
    assert reshaped_split._value.shape == (6, 2)
    assert reshaped_split.distance_axis == 0
    assert reshaped_split.max_distance == pytest.approx(arr2.max_distance * 2)

    # *args form (without tuple)
    reshaped_args = arr.reshape(n, 3)
    assert reshaped_args._value.shape == (2, 3)
    assert reshaped_args.distance_axis == 0

    reshaped_args_inferred = arr.reshape(-1, 3)
    assert reshaped_args_inferred._value.shape == (2, 3)

def test_reshape_errors(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0, 2.0], [3.0, 4.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          accountant    = accountant)
    arr2 = pnp.PrivNDArray([[1.0, 2.0, 3.0]],
                           distance      = pj.RealExpr(1),
                           distance_axis = 0,
                           accountant    = accountant)

    # axis_signature mismatch
    n2 = arr2.shape[0]
    assert isinstance(n2, pj.SensitiveDimInt)
    with pytest.raises(pj.DPError):
        arr.reshape((n2, 4))

    # scale not positive
    n = arr.shape[0]
    assert isinstance(n, pj.SensitiveDimInt)
    zero_scale = n - n
    assert isinstance(zero_scale, pj.SensitiveDimInt)
    assert zero_scale.scale == 0
    with pytest.raises(ValueError):
        arr.reshape((zero_scale, 2))

    # cannot infer -1 (PQ % known_product != 0)
    with pytest.raises(ValueError):
        arr.reshape((4, -1))

    # mixing individuals: (2, 2, N, 4) with distance_axis=2 -> (n, 16)
    arr3 = pnp.PrivNDArray(_np.arange(32).reshape(2, 2, 2, 4).astype(float),
                           distance      = pj.RealExpr(1),
                           distance_axis = 2,
                           accountant    = accountant)
    n3 = arr3.shape[2]
    assert isinstance(n3, pj.SensitiveDimInt)
    with pytest.raises(pj.DPError):
        arr3.reshape((n3, 16))


def test_privndarray_arithmetic(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0, 2.0], [3.0, 4.0]],
                          distance      = pj.RealExpr(1),
                          distance_axis = 0,
                          domain        = pnp.NDArrayDomain(value_range=(0.0, 10.0)),
                          accountant    = accountant)

    # __neg__
    neg = -arr
    assert _np.allclose(neg._value, [[-1.0, -2.0], [-3.0, -4.0]])
    assert neg.distance == arr.distance
    assert neg._domain.value_range == (-10.0, 0.0)
    assert neg.axis_signature == arr.axis_signature

    # __add__ (scalar)
    added = arr + 5
    assert _np.allclose(added._value, [[6.0, 7.0], [8.0, 9.0]])
    assert added.distance == arr.distance
    assert added._domain.value_range == (5.0, 15.0)

    # __radd__ (scalar)
    radded = 5 + arr
    assert _np.allclose(radded._value, [[6.0, 7.0], [8.0, 9.0]])

    # __sub__ (scalar)
    subbed = arr - 1
    assert _np.allclose(subbed._value, [[0.0, 1.0], [2.0, 3.0]])
    assert subbed._domain.value_range == (-1.0, 9.0)

    # __rsub__ (scalar)
    rsubbed = 10 - arr
    assert _np.allclose(rsubbed._value, [[9.0, 8.0], [7.0, 6.0]])
    assert rsubbed._domain.value_range == (0.0, 10.0)

    # __mul__ (positive scalar)
    mulled = arr * 2
    assert _np.allclose(mulled._value, [[2.0, 4.0], [6.0, 8.0]])
    assert mulled._domain.value_range == (0.0, 20.0)

    # __mul__ (negative scalar)
    mulled_neg = arr * -2
    assert _np.allclose(mulled_neg._value, [[-2.0, -4.0], [-6.0, -8.0]])
    assert mulled_neg._domain.value_range == (-20.0, 0.0)

    # __rmul__ (scalar)
    rmulled = 3 * arr
    assert _np.allclose(rmulled._value, [[3.0, 6.0], [9.0, 12.0]])

    # __truediv__ (positive scalar)
    divided = arr / 2
    assert _np.allclose(divided._value, [[0.5, 1.0], [1.5, 2.0]])
    assert divided._domain.value_range == (0.0, 5.0)

    # __truediv__ (negative scalar)
    divided_neg = arr / -2
    assert _np.allclose(divided_neg._value, [[-0.5, -1.0], [-1.5, -2.0]])
    assert divided_neg._domain.value_range == (-5.0, 0.0)

    # __truediv__ by zero
    with pytest.raises(ZeroDivisionError):
        arr / 0

    # __rtruediv__ (scalar / arr) - arr range includes 0 at boundary
    rdivided = 10 / arr
    assert _np.allclose(rdivided._value, [[10.0, 5.0], [10/3, 2.5]])
    assert rdivided._domain.value_range is None  # (0.0, 10.0) includes 0

    # __rtruediv__ with strictly positive range
    arr_positive = arr + 1  # range becomes (1.0, 11.0)
    rdivided_pos = 10 / arr_positive
    assert _np.allclose(rdivided_pos._value, [[5.0, 10/3], [2.5, 2.0]])
    assert rdivided_pos._domain.value_range == (10/11, 10.0)

    # PrivNDArray + PrivNDArray (same axis_signature)
    arr2 = arr * 2  # same axis_signature
    added_arr = arr + arr2
    assert _np.allclose(added_arr._value, [[3.0, 6.0], [9.0, 12.0]])
    assert added_arr.distance == arr.distance
    assert added_arr._domain.value_range == (0.0, 30.0)
    assert added_arr.axis_signature == arr.axis_signature

    # PrivNDArray - PrivNDArray
    subbed_arr = arr2 - arr
    assert _np.allclose(subbed_arr._value, [[1.0, 2.0], [3.0, 4.0]])
    assert subbed_arr._domain.value_range == (-10.0, 20.0)

    # PrivNDArray * PrivNDArray
    mulled_arr = arr * arr2
    assert _np.allclose(mulled_arr._value, [[2.0, 8.0], [18.0, 32.0]])
    assert mulled_arr._domain.value_range == (0.0, 200.0)

    # PrivNDArray / PrivNDArray (divisor range includes 0 -> None)
    divided_arr = arr2 / arr
    assert _np.allclose(divided_arr._value, [[2.0, 2.0], [2.0, 2.0]])
    assert divided_arr._domain.value_range is None  # arr range (0.0, 10.0) includes 0

    # PrivNDArray / PrivNDArray (strictly positive divisor)
    arr_pos = arr + 1  # range (1.0, 11.0)
    arr2_pos = arr_pos * 2  # range (2.0, 22.0)
    divided_pos = arr2_pos / arr_pos
    assert _np.allclose(divided_pos._value, [[2.0, 2.0], [2.0, 2.0]])
    assert divided_pos._domain.value_range == (2/11, 22.0)

    # axis_signature mismatch (different source arrays)
    other = pnp.PrivNDArray([[1.0, 2.0], [3.0, 4.0]],
                            distance      = pj.RealExpr(1),
                            distance_axis = 0,
                            domain        = pnp.NDArrayDomain(value_range=(0.0, 10.0)),
                            accountant    = accountant)
    with pytest.raises(pj.DPError):
        arr + other

def test_broadcast_alignment(accountant: pj.ApproxAccountant) -> None:
    arr2x2 = pnp.PrivNDArray([[1.0, 2.0],
                              [3.0, 4.0]],
                             distance      = pj.RealExpr(1),
                             distance_axis = 0,
                             domain        = pnp.NDArrayDomain(value_range=(0.0, 10.0)),
                             accountant    = accountant)

    # Fail: (2,) pads to (1, 2) -> int(1) vs SensitiveDimInt at position 0
    row_max_flat = arr2x2.max(axis=1, keepdims=False)
    with pytest.raises(pj.DPError):
        arr2x2 + row_max_flat

    # OK: keepdims=True -> (2, 1) aligns with (2, 2)
    row_max_kept = arr2x2.max(axis=1, keepdims=True)
    result_2x2 = arr2x2 - row_max_kept
    assert result_2x2._value.shape == (2, 2)
    assert _np.allclose(result_2x2._value, [[-1.0, 0.0], [-1.0, 0.0]])

    arr3d_222 = pnp.PrivNDArray([[[1.0, 2.0], [3.0, 4.0]],
                                 [[5.0, 6.0], [7.0, 8.0]]],
                                distance      = pj.RealExpr(1),
                                distance_axis = 0,
                                domain        = pnp.NDArrayDomain(value_range=(0.0, 10.0)),
                                accountant    = accountant)

    # Fail: (2, 2) pads to (1, 2, 2) -> int(1) vs SensitiveDimInt at position 0
    reduced_22 = arr3d_222.max(axis=2, keepdims=False)
    with pytest.raises(pj.DPError):
        arr3d_222 + reduced_22

    # OK: keepdims=True -> (2, 2, 1) aligns with (2, 2, 2)
    reduced_kept = arr3d_222.max(axis=2, keepdims=True)
    result_222 = arr3d_222 - reduced_kept
    assert result_222._value.shape == (2, 2, 2)
    assert _np.allclose(result_222._value, [[[-1.0, 0.0], [-1.0, 0.0]],
                                            [[-1.0, 0.0], [-1.0, 0.0]]])
