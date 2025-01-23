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
    return prisoner.map(lambda x: laplace_mechanism(x, eps))

@laplace_mechanism.register(remote=False) # type: ignore
def _(prisoner: SensitiveDataFrame, eps: floating) -> _pd.DataFrame:
    return prisoner.map(lambda x: laplace_mechanism(x, eps))

# TODO: add test
@overload
def exponential_mechanism(scores: list[SensitiveInt | SensitiveFloat], eps: float) -> int: ...
@overload
def exponential_mechanism(scores: dict[T, SensitiveInt | SensitiveFloat], eps: float) -> T: ...

def exponential_mechanism(scores: list[SensitiveInt | SensitiveFloat] | dict[Any, SensitiveInt | SensitiveFloat], eps: float) -> Any:
    if len(scores) == 0:
        raise ValueError("scores must have at least one element.")

    if isinstance(scores, dict):
        keys = list(scores.keys())
        values = list(scores.values())
    elif isinstance(scores, list):
        keys = list(range(len(scores)))
        values = scores
    else:
        raise ValueError("scores must be a list or dict.")

    for v in values:
        if not isinstance(v, (SensitiveInt, SensitiveFloat)):
            raise TypeError("values of scores must be SensitiveInt or SensitiveFloat.")

    sensitivity = max([v.distance.max() for v in values]) # type: ignore[type-var]

    if sensitivity <= 0:
        raise DPError(f"Invalid sensitivity ({sensitivity})")

    if eps <= 0:
        raise DPError(f"Invalid epsilon ({eps})")

    # create a dummy prisoner to propagate budget consumption to all prisoners
    prisoner_dummy = Prisoner(0, values[0].distance, parents=values)
    prisoner_dummy.consume_privacy_budget(eps)

    exponents = [eps * v._value / sensitivity / 2 for v in values]
    # to prevent too small or large values (-> 0 or inf)
    M: float = _np.max(exponents) # type: ignore[arg-type]
    p = [_np.exp(x - M) for x in exponents]
    p /= sum(p)
    return _np.random.choice(keys, p=p)
