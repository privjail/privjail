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
from .accountants import PureAccountant, ApproxAccountant, zCDPAccountant
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat
from .pandas import SensitiveSeries, SensitiveDataFrame
from .pandas.util import ElementType, Index, MultiIndex, pack_pandas_index
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

def assert_delta(delta: floating) -> None:
    if delta <= 0 or 1.0 < delta:
        raise DPError(f"Invalid delta ({delta})")

def assert_rho(rho: floating) -> None:
    if rho <= 0:
        raise DPError(f"Invalid rho ({rho})")

# TODO: serialization/deserialization for numpy arrays might be slow
@egrpc.dataclass
class FloatSeriesBuf:
    values : list[float]
    index  : Index | MultiIndex
    name   : ElementType | None

@egrpc.dataclass
class FloatDataFrameBuf:
    values  : list[list[float]]
    index   : Index | MultiIndex
    columns : Index | MultiIndex

def laplace_mechanism(prisoner: Any, eps: floating) -> float | _pd.Series | _pd.DataFrame: # type: ignore[type-arg]
    result = laplace_mechanism_impl(prisoner, float(eps))

    if isinstance(result, float):
        return result
    if isinstance(result, FloatSeriesBuf):
        return _pd.Series(result.values, index=result.index.to_pandas(), name=result.name)
    elif isinstance(result, FloatDataFrameBuf):
        return _pd.DataFrame(result.values, index=result.index.to_pandas(), columns=result.columns.to_pandas())
    else:
        raise Exception

@egrpc.multifunction
def laplace_mechanism_impl(prisoner: SensitiveInt | SensitiveFloat, eps: float) -> float:
    assert_eps(eps)

    sensitivity = float(prisoner.distance.max())
    assert_sensitivity(sensitivity)

    result = float(_np.random.laplace(loc=prisoner._value, scale=sensitivity / eps))

    if isinstance(prisoner.accountant, PureAccountant):
        prisoner.accountant.spend(eps)
    elif isinstance(prisoner.accountant, ApproxAccountant):
        prisoner.accountant.spend((eps, 0.0))
    else:
        raise RuntimeError

    return result

@laplace_mechanism_impl.register
def _(prisoner: SensitiveSeries[realnum], eps: float) -> FloatSeriesBuf:
    assert_eps(eps)

    if prisoner._distance_group == "val":
        eps_each = eps / len(prisoner)
        scales = []
        assert isinstance(prisoner._distance_per_val, list)
        for distance in prisoner._distance_per_val:
            sensitivity = float(distance.max())
            assert_sensitivity(sensitivity)
            scales.append(sensitivity / eps_each)
        data = _np.random.laplace(loc=prisoner._value, scale=scales)
    else:
        sensitivity = float(prisoner.distance.max())
        assert_sensitivity(sensitivity)
        data = _np.random.laplace(loc=prisoner._value, scale=sensitivity / eps)

    if isinstance(prisoner.accountant, PureAccountant):
        prisoner.accountant.spend(eps)
    elif isinstance(prisoner.accountant, ApproxAccountant):
        prisoner.accountant.spend((eps, 0.0))
    else:
        raise RuntimeError

    return FloatSeriesBuf(data.tolist(), pack_pandas_index(prisoner.index), prisoner.name)

@laplace_mechanism_impl.register
def _(prisoner: SensitiveDataFrame, eps: float) -> FloatDataFrameBuf:
    assert_eps(eps)

    if prisoner._distance_group == "ser":
        eps_each = eps / len(prisoner)
        scales = []
        assert isinstance(prisoner._distance_per_ser, list)
        for distance in prisoner._distance_per_ser:
            sensitivity = float(distance.max())
            assert_sensitivity(sensitivity)
            scales.append(sensitivity / eps_each)
        data = _np.random.laplace(loc=prisoner._value, scale=scales)
    else:
        sensitivity = float(prisoner.distance.max())
        assert_sensitivity(sensitivity)
        data = _np.random.laplace(loc=prisoner._value, scale=sensitivity / eps)

    if isinstance(prisoner.accountant, PureAccountant):
        prisoner.accountant.spend(eps)
    elif isinstance(prisoner.accountant, ApproxAccountant):
        prisoner.accountant.spend((eps, 0.0))
    else:
        raise RuntimeError

    return FloatDataFrameBuf(data.tolist(), pack_pandas_index(prisoner.index), pack_pandas_index(prisoner.columns))

def sigma_from_eps_delta(sensitivity: float, eps: float, delta: float) -> float:
    # The Algorithmic Foundations of Differential Privacy
    # https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
    # Theorem 3.22
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps

def sigma_from_rho(sensitivity: float, rho: float) -> float:
    # Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds
    # https://arxiv.org/pdf/1605.02065
    # Proposition 1.6
    return sensitivity / math.sqrt(2.0 * rho)

def gaussian_mechanism(prisoner : Any,
                       *,
                       eps      : floating | None = None,
                       delta    : floating | None = None,
                       rho      : floating | None = None,
                       ) -> float | _pd.Series | _pd.DataFrame: # type: ignore[type-arg]
    result = gaussian_mechanism_impl(prisoner,
                                     eps   = float(eps) if eps is not None else None,
                                     delta = float(delta) if delta is not None else None,
                                     rho   = float(rho) if rho is not None else None)

    if isinstance(result, float):
        return result
    if isinstance(result, FloatSeriesBuf):
        return _pd.Series(result.values, index=result.index.to_pandas(), name=result.name)
    elif isinstance(result, FloatDataFrameBuf):
        return _pd.DataFrame(result.values, index=result.index.to_pandas(), columns=result.columns.to_pandas())
    else:
        raise Exception

@egrpc.multifunction
def gaussian_mechanism_impl(prisoner : SensitiveInt | SensitiveFloat,
                            *,
                            eps      : float | None = None,
                            delta    : float | None = None,
                            rho      : float | None = None,
                            ) -> float:
    sensitivity = float(prisoner.distance.max())
    assert_sensitivity(sensitivity)

    if isinstance(prisoner.accountant, PureAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxAccountant):
        assert eps is not None
        assert delta is not None
        assert_eps(eps)
        assert_delta(delta)

        sigma = sigma_from_eps_delta(sensitivity, eps, delta)
        budget = (eps, delta)

    elif isinstance(prisoner.accountant, zCDPAccountant):
        assert rho is not None
        assert_rho(rho)

        sigma = sigma_from_rho(sensitivity, rho)
        budget = rho

    else:
        raise RuntimeError

    result = float(_np.random.normal(loc=prisoner._value, scale=sigma))

    prisoner.accountant.spend(budget)

    return result

@gaussian_mechanism_impl.register
def _(prisoner : SensitiveSeries[realnum],
      *,
      eps      : float | None = None,
      delta    : float | None = None,
      rho      : float | None = None,
      ) -> FloatSeriesBuf:
    if isinstance(prisoner.accountant, PureAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxAccountant):
        assert eps is not None
        assert delta is not None
        assert_eps(eps)
        assert_delta(delta)

        if prisoner._distance_group == "val":
            eps_each = eps / len(prisoner)
            delta_each = delta / len(prisoner)
            scales = []
            assert isinstance(prisoner._distance_per_val, list)
            for distance in prisoner._distance_per_val:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                sigma = sigma_from_eps_delta(sensitivity, eps_each, delta_each)
                scales.append(sigma)
            data = _np.random.normal(loc=prisoner._value, scale=scales)
        else:
            # L1 sensitivity = L2 sensitivity for groupby-count queries
            sensitivity = float(prisoner.distance.max())
            assert_sensitivity(sensitivity)
            sigma = sigma_from_eps_delta(sensitivity, eps, delta)
            data = _np.random.normal(loc=prisoner._value, scale=sigma)

        budget = (eps, delta)

    elif isinstance(prisoner.accountant, zCDPAccountant):
        assert rho is not None
        assert_rho(rho)

        if prisoner._distance_group == "val":
            rho_each = rho / len(prisoner)
            scales = []
            assert isinstance(prisoner._distance_per_val, list)
            for distance in prisoner._distance_per_val:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                sigma = sigma_from_rho(sensitivity, rho_each)
                scales.append(sigma)
            data = _np.random.normal(loc=prisoner._value, scale=scales)
        else:
            # L1 sensitivity = L2 sensitivity for groupby-count queries
            sensitivity = float(prisoner.distance.max())
            assert_sensitivity(sensitivity)
            sigma = sigma_from_rho(sensitivity, rho)
            data = _np.random.normal(loc=prisoner._value, scale=sigma)

        budget = rho

    else:
        raise RuntimeError

    prisoner.accountant.spend(budget)

    return FloatSeriesBuf(data.tolist(), pack_pandas_index(prisoner.index), prisoner.name)

@gaussian_mechanism_impl.register
def _(prisoner : SensitiveDataFrame,
      *,
      eps      : float | None = None,
      delta    : float | None = None,
      rho      : float | None = None,
      ) -> FloatDataFrameBuf:
    if isinstance(prisoner.accountant, PureAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxAccountant):
        assert eps is not None
        assert delta is not None
        assert_eps(eps)
        assert_delta(delta)

        if prisoner._distance_group == "ser":
            eps_each = eps / len(prisoner)
            delta_each = delta / len(prisoner)
            scales = []
            assert isinstance(prisoner._distance_per_ser, list)
            for distance in prisoner._distance_per_ser:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                sigma = sigma_from_eps_delta(sensitivity, eps_each, delta_each)
                scales.append(sigma)
            data = _np.random.normal(loc=prisoner._value, scale=scales)
        else:
            # L1 sensitivity = L2 sensitivity for groupby-count queries
            sensitivity = float(prisoner.distance.max())
            assert_sensitivity(sensitivity)
            sigma = sigma_from_eps_delta(sensitivity, eps, delta)
            data = _np.random.normal(loc=prisoner._value, scale=sigma)

        budget = (eps, delta)

    elif isinstance(prisoner.accountant, zCDPAccountant):
        assert rho is not None
        assert_rho(rho)

        if prisoner._distance_group == "ser":
            rho_each = rho / len(prisoner)
            scales = []
            assert isinstance(prisoner._distance_per_ser, list)
            for distance in prisoner._distance_per_ser:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                sigma = sigma_from_rho(sensitivity, rho_each)
                scales.append(sigma)
            data = _np.random.normal(loc=prisoner._value, scale=scales)
        else:
            # L1 sensitivity = L2 sensitivity for groupby-count queries
            sensitivity = float(prisoner.distance.max())
            assert_sensitivity(sensitivity)
            sigma = sigma_from_rho(sensitivity, rho)
            data = _np.random.normal(loc=prisoner._value, scale=sigma)

        budget = rho

    else:
        raise RuntimeError

    prisoner.accountant.spend(budget)

    return FloatDataFrameBuf(data.tolist(), pack_pandas_index(prisoner.index), pack_pandas_index(prisoner.columns))

@egrpc.function
def exponential_mechanism(scores : Sequence[SensitiveInt | SensitiveFloat],
                          *,
                          eps    : float | None = None,
                          rho    : float | None = None,
                          ) -> int:
    if len(scores) == 0:
        raise ValueError("scores must have at least one element.")

    sensitivity = max([v.distance.max() for v in scores]) # type: ignore[type-var]
    assert_sensitivity(sensitivity)

    # create a dummy prisoner to propagate budget consumption to all prisoners
    prisoner_dummy = Prisoner(0, scores[0].distance, parents=scores)

    if isinstance(prisoner_dummy.accountant, PureAccountant):
        assert eps is not None
        assert_eps(eps)
        budget = eps

    elif isinstance(prisoner_dummy.accountant, ApproxAccountant):
        assert eps is not None
        assert_eps(eps)
        budget = (eps, 0.0)

    elif isinstance(prisoner_dummy.accountant, zCDPAccountant):
        # Bounding, Concentrating, and Truncating: Unifying Privacy Loss Composition for Data Analytics
        # https://arxiv.org/pdf/2004.07223
        if eps is not None and rho is not None:
            raise ValueError("only one of eps and rho should be specified")
        elif eps is not None:
            assert_eps(eps)
            rho = eps * eps / 8.0
        elif rho is not None:
            assert_rho(rho)
            eps = math.sqrt(8.0 * rho)
        else:
            raise ValueError("eps or rho should be specified")

        budget = rho

    else:
        raise RuntimeError

    exponents = [eps * s._value / sensitivity / 2 for s in scores]

    # to prevent too small or large values (-> 0 or inf)
    M: float = _np.max(exponents) # type: ignore[arg-type]
    p = [_np.exp(x - M) for x in exponents]
    p /= sum(p)
    result = _np.random.choice(len(scores), p=p)

    prisoner_dummy.accountant.spend(budget)

    return result

def argmax(args : Sequence[SensitiveInt | SensitiveFloat],
           *,
           eps  : floating | None = None,
           rho  : floating | None = None,
           mech : str             = "exp",
           ) -> int:
    if mech == "exp":
        return exponential_mechanism(args,
                                     eps = float(eps) if eps is not None else None,
                                     rho = float(rho) if rho is not None else None)
    else:
        raise ValueError(f"Unknown DP mechanism: '{mech}'")

def argmin(args : Sequence[SensitiveInt | SensitiveFloat],
           *,
           eps  : floating | None = None,
           rho  : floating | None = None,
           mech : str             = "exp",
           ) -> int:
    args_negative = [-x for x in args]
    if mech == "exp":
        return exponential_mechanism(args_negative,
                                     eps = float(eps) if eps is not None else None,
                                     rho = float(rho) if rho is not None else None)
    else:
        raise ValueError(f"Unknown DP mechanism: '{mech}'")
