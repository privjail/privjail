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

from typing import TypeVar, Sequence, Any
import math

import numpy as _np
import pandas as _pd

from .util import DPError, floating, realnum
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat
from .pandas import ElementType, SensitiveSeries, SensitiveDataFrame
from . import egrpc

T = TypeVar("T")

def assert_sensitivity(sensitivity: realnum) -> None:
    if sensitivity == math.inf:
        raise DPError(f"Unbounded sensitivity")

    if sensitivity <= 0:
        raise DPError(f"Invalid sensitivity ({sensitivity})")

def assert_eps(eps: floating) -> None:
    if eps <= 0:
        raise DPError(f"Invalid epsilon ({eps})")

# TODO: serialization/deserialization for numpy arrays might be slow
@egrpc.dataclass
class FloatSeriesBuf:
    values : list[float]
    index  : list[ElementType]
    name   : ElementType | None

@egrpc.dataclass
class FloatDataFrameBuf:
    values  : list[list[float]]
    index   : list[ElementType]
    columns : list[ElementType]

def laplace_mechanism(prisoner: Any, eps: floating) -> float | _pd.Series | _pd.DataFrame: # type: ignore[type-arg]
    result = laplace_mechanism_impl(prisoner, eps)

    if isinstance(result, float):
        return result
    if isinstance(result, FloatSeriesBuf):
        return _pd.Series(result.values, index=result.index, name=result.name)
    elif isinstance(result, FloatDataFrameBuf):
        return _pd.DataFrame(result.values, index=result.index, columns=result.columns)
    else:
        raise Exception

@egrpc.multifunction
def laplace_mechanism_impl(prisoner: SensitiveInt | SensitiveFloat, eps: floating) -> float:
    assert_eps(eps)

    sensitivity = prisoner.distance.max()
    assert_sensitivity(sensitivity)

    result = float(_np.random.laplace(loc=prisoner._value, scale=sensitivity / eps))

    prisoner.consume_privacy_budget(float(eps))

    return result

@laplace_mechanism_impl.register
def _(prisoner: SensitiveSeries[realnum], eps: floating) -> FloatSeriesBuf:
    assert_eps(eps)

    if prisoner._distance_group == "val":
        eps_each = eps / len(prisoner)
        scales = []
        assert isinstance(prisoner._distance_per_val, list)
        for distance in prisoner._distance_per_val:
            sensitivity = distance.max()
            assert_sensitivity(sensitivity)
            scales.append(float(sensitivity / eps_each))
        data = _np.random.laplace(loc=prisoner._value, scale=scales)
    else:
        sensitivity = prisoner.distance.max()
        assert_sensitivity(sensitivity)
        data = _np.random.laplace(loc=prisoner._value, scale=sensitivity / eps)

    prisoner.consume_privacy_budget(float(eps))

    return FloatSeriesBuf(data.tolist(), prisoner.index.tolist(), prisoner.name)

@laplace_mechanism_impl.register
def _(prisoner: SensitiveDataFrame, eps: floating) -> FloatDataFrameBuf:
    assert_eps(eps)

    if prisoner._distance_group == "ser":
        eps_each = eps / len(prisoner)
        scales = []
        assert isinstance(prisoner._distance_per_ser, list)
        for distance in prisoner._distance_per_ser:
            sensitivity = distance.max()
            assert_sensitivity(sensitivity)
            scales.append(float(sensitivity / eps_each))
        data = _np.random.laplace(loc=prisoner._value, scale=scales)
    else:
        sensitivity = prisoner.distance.max()
        assert_sensitivity(sensitivity)
        data = _np.random.laplace(loc=prisoner._value, scale=sensitivity / eps)

    prisoner.consume_privacy_budget(float(eps))

    return FloatDataFrameBuf(data.tolist(), prisoner.index.tolist(), prisoner.columns.tolist())

# @laplace_mechanism.register(remote=False) # type: ignore
# def _(prisoner: Series[realnum], eps: floating) -> _pd.Series: # type: ignore[type-arg]
#     total_distance = prisoner.max_distance()
#     return prisoner.map(lambda x: laplace_mechanism(x, eps * x.max_distance / total_distance))

# @laplace_mechanism.register(remote=False) # type: ignore
# def _(prisoner: DataFrame, eps: floating) -> _pd.DataFrame:
#     total_distance = prisoner.max_distance()
#     return prisoner.map(lambda x: laplace_mechanism(x, eps * x.max_distance / total_distance))

@egrpc.function
def exponential_mechanism(scores: Sequence[SensitiveInt | SensitiveFloat], eps: floating) -> int:
    if len(scores) == 0:
        raise ValueError("scores must have at least one element.")

    assert_eps(eps)

    sensitivity = max([v.distance.max() for v in scores]) # type: ignore[type-var]
    assert_sensitivity(sensitivity)

    exponents = [eps * s._value / sensitivity / 2 for s in scores]
    # to prevent too small or large values (-> 0 or inf)
    M: float = _np.max(exponents) # type: ignore[arg-type]
    p = [_np.exp(x - M) for x in exponents]
    p /= sum(p)
    result = _np.random.choice(len(scores), p=p)

    # create a dummy prisoner to propagate budget consumption to all prisoners
    prisoner_dummy = Prisoner(0, scores[0].distance, parents=scores)
    prisoner_dummy.consume_privacy_budget(float(eps))

    return result

def argmax(args: Sequence[SensitiveInt | SensitiveFloat], eps: floating, mech: str = "exponential") -> int:
    if mech == "exponential":
        return exponential_mechanism(args, eps)
    else:
        raise ValueError(f"Unknown DP mechanism: '{mech}'")

def argmin(args: Sequence[SensitiveInt | SensitiveFloat], eps: floating, mech: str = "exponential") -> int:
    args_negative = [-x for x in args]
    if mech == "exponential":
        return exponential_mechanism(args_negative, eps)
    else:
        raise ValueError(f"Unknown DP mechanism: '{mech}'")
