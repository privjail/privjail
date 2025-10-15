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
                          distance   = pj.RealExpr(1),
                          accountant = accountant)

    assert arr.ndim == 2
    assert isinstance(arr.shape[0], pj.SensitiveInt)
    assert all(isinstance(x, int) for x in arr.shape[1:])
    assert arr.shape[0]._value == 4
    assert arr.shape[1] == 3

    bound = 5.0
    clipped = arr.clip_norm(bound=bound, ord=2)

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

def test_clip_norm_scalar_rows(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([1.0, -2.5, 0.5, 4.0, -6.0],
                          distance   = pj.RealExpr(1),
                          accountant = accountant)

    bound = 2.0
    clipped = arr.clip_norm(bound=bound, ord=2)

    assert clipped._value.tolist() == pytest.approx([1.0, -2.0, 0.5, 2.0, -2.0])
    assert arr._value.tolist() == [1.0, -2.5, 0.5, 4.0, -6.0]

def test_clip_norm_matrix_rows(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[ 3.0, 4.0,  0.0],
                           [ 0.0, 0.0,  0.0],
                           [ 6.0, 8.0,  2.0],
                           [-1.0, 2.0, -2.0]],
                          distance   = pj.RealExpr(1),
                          accountant = accountant)

    bound = 5.0
    clipped = arr.clip_norm(bound=bound, ord=2)

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
                          distance   = pj.RealExpr(1),
                          accountant = accountant)

    bound = 6.0
    clipped = arr.clip_norm(bound=bound, ord=2)

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

def test_sum_returns_sensitive_float(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([1.0, -1.5, 0.5, 2.0],
                          distance   = pj.RealExpr(1),
                          accountant = accountant)

    bound = 2.0
    clipped = arr.clip_norm(bound=bound, ord=1)
    total = clipped.sum(axis=0)

    assert isinstance(total, pj.SensitiveFloat)
    assert total._value == pytest.approx(clipped._value.sum())
    assert total.max_distance == pytest.approx(bound)

def test_sum_returns_sensitive_ndarray(accountant: pj.ApproxAccountant) -> None:
    arr = pnp.PrivNDArray([[1.0,  2.0, -1.5],
                           [0.5, -4.0,  3.0],
                           [2.0,  1.0, -0.5]],
                          distance   = pj.RealExpr(1),
                          accountant = accountant)

    bound = 3.0
    clipped = arr.clip_norm(bound=bound, ord=1)
    summed = clipped.sum(axis=0)

    assert isinstance(summed, pnp.SensitiveNDArray)
    assert summed.norm_type == "l1"
    assert summed.shape == (3,)

    expected = _np.asarray(clipped._value.sum(axis=0), dtype=float)
    assert summed._value.tolist() == pytest.approx(expected.tolist())
    assert summed.max_distance == pytest.approx(bound)

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
