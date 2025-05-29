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

from typing import Any, TypeVar, overload

import numpy as _np
import pandas as _pd

from .util import DPError, floating, realnum
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat
from .pandas import SensitiveSeries, SensitiveDataFrame
from . import egrpc

T = TypeVar("T")

@egrpc.multifunction
def laplace_mechanism(prisoner: SensitiveInt | SensitiveFloat, eps: floating) -> float:
    sensitivity = prisoner.distance.max()

    if sensitivity <= 0:
        raise DPError(f"Invalid sensitivity ({sensitivity})")

    if eps <= 0:
        raise DPError(f"Invalid epsilon ({eps})")

    prisoner.consume_privacy_budget(float(eps))

    return float(_np.random.laplace(loc=prisoner._value, scale=sensitivity / eps))

@laplace_mechanism.register(remote=False) # type: ignore
def _(prisoner: SensitiveSeries[realnum], eps: floating) -> _pd.Series: # type: ignore[type-arg]
    total_distance = prisoner.max_distance()
    return prisoner.map(lambda x: laplace_mechanism(x, eps * x.max_distance / total_distance))

@laplace_mechanism.register(remote=False) # type: ignore
def _(prisoner: SensitiveDataFrame, eps: floating) -> _pd.DataFrame:
    total_distance = prisoner.max_distance()
    return prisoner.map(lambda x: laplace_mechanism(x, eps * x.max_distance / total_distance))

# TODO: add test
@overload
def exponential_mechanism(scores: list[SensitiveInt | SensitiveFloat], eps: float) -> int: ...
@overload
def exponential_mechanism(scores: dict[T, SensitiveInt | SensitiveFloat], eps: float) -> T: ...

def exponential_mechanism(scores: list[SensitiveInt | SensitiveFloat] | dict[Any, SensitiveInt | SensitiveFloat], eps: float) -> Any:
    if isinstance(scores, dict):
        idx = _exponential_mechanism_impl(list(scores.values()), eps)
        return list(scores.keys())[idx]
    elif isinstance(scores, list):
        return _exponential_mechanism_impl(scores, eps)
    else:
        raise ValueError("scores must be a list or dict.")

@egrpc.function
def _exponential_mechanism_impl(scores: list[SensitiveInt | SensitiveFloat], eps: float) -> int:
    if len(scores) == 0:
        raise ValueError("scores must have at least one element.")

    sensitivity = max([v.distance.max() for v in scores]) # type: ignore[type-var]

    if sensitivity <= 0:
        raise DPError(f"Invalid sensitivity ({sensitivity})")

    if eps <= 0:
        raise DPError(f"Invalid epsilon ({eps})")

    # create a dummy prisoner to propagate budget consumption to all prisoners
    prisoner_dummy = Prisoner(0, scores[0].distance, parents=scores)
    prisoner_dummy.consume_privacy_budget(eps)

    exponents = [eps * s._value / sensitivity / 2 for s in scores]
    # to prevent too small or large values (-> 0 or inf)
    M: float = _np.max(exponents) # type: ignore[arg-type]
    p = [_np.exp(x - M) for x in exponents]
    p /= sum(p)
    return _np.random.choice(len(scores), p=p)
