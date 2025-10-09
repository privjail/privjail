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

from typing import TypeVar, Sequence, Any, overload
import math

import numpy as _np
import pandas as _pd
import numpy.typing as _npt

from .util import DPError, floating, realnum, ElementType
from .accountants import BudgetType, PureAccountant, ApproxAccountant, zCDPAccountant
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat
from .numpy import SensitiveNDArray
from .pandas import SensitiveSeries, SensitiveDataFrame
from .pandas.util import Index, MultiIndex, pack_pandas_index
from . import egrpc

T = TypeVar("T")

def assert_sensitivity(sensitivity: realnum) -> None:
    if sensitivity == math.inf:
        raise DPError("Unbounded sensitivity")

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

def assert_laplace_scale(scale: floating) -> None:
    if scale <= 0:
        raise DPError(f"Invalid scale ({scale})")

def assert_gaussian_scale(scale: floating) -> None:
    if scale <= 0:
        raise DPError(f"Invalid scale ({scale})")

def resolve_laplace_params(sensitivity : float,
                           *,
                           eps         : float | None = None,
                           scale       : float | None = None,
                           ) -> tuple[float, float]:
    if eps is None and scale is None:
        raise ValueError("Either eps or scale must be specified.")

    elif eps is not None and scale is not None:
        raise ValueError("eps and scale cannot be specified simultaneously.")

    elif eps is not None:
        assert_eps(eps)
        resolved_scale = sensitivity / eps
        assert_laplace_scale(resolved_scale)
        return eps, resolved_scale

    elif scale is not None:
        assert_laplace_scale(scale)
        resolved_eps = sensitivity / scale
        assert_eps(resolved_eps)
        return resolved_eps, scale

    else:
        raise Exception

def sigma_from_eps_delta(sensitivity: float, eps: float, delta: float) -> float:
    # The Algorithmic Foundations of Differential Privacy
    # https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
    # Theorem 3.22
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / eps

def eps_from_sigma_delta(sensitivity: float, sigma: float, delta: float) -> float:
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / sigma

def sigma_from_rho(sensitivity: float, rho: float) -> float:
    # Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds
    # https://arxiv.org/pdf/1605.02065
    # Proposition 1.6
    return sensitivity / math.sqrt(2.0 * rho)

def rho_from_sigma(sensitivity: float, sigma: float) -> float:
    return sensitivity * sensitivity / (2.0 * sigma * sigma)

def resolve_gaussian_params_approx(sensitivity : float,
                                   *,
                                   eps         : float | None = None,
                                   delta       : float | None = None,
                                   scale       : float | None = None,
                                   ) -> tuple[float, float]:
    if delta is None:
        raise ValueError("delta must be specified when using the Gaussian mechanism under approx DP.")

    elif eps is None and scale is None:
        raise ValueError("Either eps or scale must be specified.")

    elif eps is not None and scale is not None:
        raise ValueError("eps and scale cannot be specified simultaneously.")

    elif eps is not None:
        assert_eps(eps)
        resolved_scale = sigma_from_eps_delta(sensitivity, eps, delta)
        assert_gaussian_scale(resolved_scale)
        return eps, resolved_scale

    elif scale is not None:
        assert_gaussian_scale(scale)
        resolved_eps = eps_from_sigma_delta(sensitivity, scale, delta)
        assert_eps(resolved_eps)
        return resolved_eps, scale

    else:
        raise Exception

def resolve_gaussian_params_zcdp(sensitivity : float,
                                 *,
                                 rho         : float | None = None,
                                 scale       : float | None = None,
                                 ) -> tuple[float, float]:
    if rho is None and scale is None:
        raise ValueError("Either rho or scale must be specified.")

    elif rho is not None and scale is not None:
        raise ValueError("rho and scale cannot be specified simultaneously.")

    elif rho is not None:
        assert_rho(rho)
        resolved_scale = sigma_from_rho(sensitivity, rho)
        assert_gaussian_scale(resolved_scale)
        return rho, resolved_scale

    elif scale is not None:
        assert_gaussian_scale(scale)
        resolved_rho = rho_from_sigma(sensitivity, scale)
        assert_rho(resolved_rho)
        return resolved_rho, scale

    else:
        raise Exception

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

@egrpc.dataclass
class FloatArrayBuf:
    values : list[float]
    shape  : tuple[int, ...]

@overload
def laplace_mechanism(prisoner : SensitiveInt | SensitiveFloat,
                      *,
                      eps      : floating | None = ...,
                      scale    : floating | None = ...,
                      ) -> float: ...

@overload
def laplace_mechanism(prisoner : SensitiveSeries[Any],
                      *,
                      eps      : floating | None = ...,
                      scale    : floating | None = ...,
                      ) -> _pd.Series: ... # type: ignore[type-arg]

@overload
def laplace_mechanism(prisoner : SensitiveDataFrame,
                      *,
                      eps      : floating | None = ...,
                      scale    : floating | None = ...,
                      ) -> _pd.DataFrame: ...

@overload
def laplace_mechanism(prisoner : SensitiveNDArray,
                      *,
                      eps      : floating | None = ...,
                      scale    : floating | None = ...,
                      ) -> _npt.NDArray[Any]: ...

def laplace_mechanism(prisoner : Any,
                      *,
                      eps      : floating | None = None,
                      scale    : floating | None = None,
                      ) -> float | _pd.Series | _pd.DataFrame | _npt.NDArray[Any]: # type: ignore[type-arg]
    if eps is None and scale is None:
        raise ValueError("Either eps or scale must be specified.")

    elif eps is not None and scale is not None:
        raise ValueError("eps and scale cannot be specified simultaneously.")

    result = laplace_mechanism_impl(prisoner,
                                    eps   = float(eps)   if eps   is not None else None,
                                    scale = float(scale) if scale is not None else None)

    if isinstance(result, float):
        return result
    if isinstance(result, FloatSeriesBuf):
        return _pd.Series(result.values, index=result.index.to_pandas(), name=result.name)
    elif isinstance(result, FloatDataFrameBuf):
        return _pd.DataFrame(result.values, index=result.index.to_pandas(), columns=result.columns.to_pandas())
    elif isinstance(result, FloatArrayBuf):
        return _np.asarray(result.values).reshape(result.shape)
    else:
        raise Exception

@egrpc.multifunction
def laplace_mechanism_impl(prisoner : SensitiveInt | SensitiveFloat,
                           *,
                           eps      : float | None = None,
                           scale    : float | None = None,
                           ) -> float:
    sensitivity = float(prisoner.distance.max())
    assert_sensitivity(sensitivity)

    resolved_eps, resolved_scale = resolve_laplace_params(sensitivity, eps=eps, scale=scale)

    result = float(_np.random.laplace(loc=prisoner._value, scale=resolved_scale))

    if isinstance(prisoner.accountant, PureAccountant):
        prisoner.accountant.spend(resolved_eps)
    elif isinstance(prisoner.accountant, ApproxAccountant):
        prisoner.accountant.spend((resolved_eps, 0.0))
    else:
        raise RuntimeError

    return result

@laplace_mechanism_impl.register
def _(prisoner : SensitiveSeries[realnum],
      *,
      eps      : float | None = None,
      scale    : float | None = None,
      ) -> FloatSeriesBuf:
    if prisoner._distance_group_axes == (0,):
        assert isinstance(prisoner._partitioned_distances, list)

        if eps is not None:
            assert_eps(eps)
            eps_each = eps / len(prisoner)
            scales = []
            for distance in prisoner._partitioned_distances:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                _, resolved_scale = resolve_laplace_params(sensitivity, eps=eps_each)
                scales.append(resolved_scale)
            data = _np.random.laplace(loc=prisoner._value, scale=scales)
            spent_eps = eps

        elif scale is not None:
            assert_laplace_scale(scale)
            total_eps = 0.0
            for distance in prisoner._partitioned_distances:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                resolved_eps, _ = resolve_laplace_params(sensitivity, scale=scale)
                total_eps += resolved_eps
            data = _np.random.laplace(loc=prisoner._value, scale=scale)
            spent_eps = total_eps

        else:
            raise Exception

    else:
        sensitivity = float(prisoner.distance.max())
        resolved_eps, resolved_scale = resolve_laplace_params(sensitivity, eps=eps, scale=scale)
        data = _np.random.laplace(loc=prisoner._value, scale=resolved_scale)
        spent_eps = resolved_eps

    if isinstance(prisoner.accountant, PureAccountant):
        prisoner.accountant.spend(spent_eps)
    elif isinstance(prisoner.accountant, ApproxAccountant):
        prisoner.accountant.spend((spent_eps, 0.0))
    else:
        raise RuntimeError

    return FloatSeriesBuf(data.tolist(), pack_pandas_index(prisoner.index), prisoner.name)

@laplace_mechanism_impl.register
def _(prisoner : SensitiveDataFrame,
      *,
      eps      : float | None = None,
      scale    : float | None = None,
      ) -> FloatDataFrameBuf:
    if prisoner._distance_group_axes == (1,):
        assert isinstance(prisoner._partitioned_distances, list)

        if eps is not None:
            assert_eps(eps)
            eps_each = eps / len(prisoner)
            scales = []
            for distance in prisoner._partitioned_distances:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                _, resolved_scale = resolve_laplace_params(sensitivity, eps=eps_each)
                scales.append(resolved_scale)
            data = _np.random.laplace(loc=prisoner._value, scale=scales)
            spent_eps = eps

        elif scale is not None:
            assert_laplace_scale(scale)
            total_eps = 0.0
            for distance in prisoner._partitioned_distances:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                resolved_eps, _ = resolve_laplace_params(sensitivity, scale=scale)
                total_eps += resolved_eps
            data = _np.random.laplace(loc=prisoner._value, scale=scale)
            spent_eps = total_eps

        else:
            raise Exception

    else:
        sensitivity = float(prisoner.distance.max())
        resolved_eps, resolved_scale = resolve_laplace_params(sensitivity, eps=eps, scale=scale)
        data = _np.random.laplace(loc=prisoner._value, scale=resolved_scale)
        spent_eps = resolved_eps

    if isinstance(prisoner.accountant, PureAccountant):
        prisoner.accountant.spend(spent_eps)
    elif isinstance(prisoner.accountant, ApproxAccountant):
        prisoner.accountant.spend((spent_eps, 0.0))
    else:
        raise RuntimeError

    return FloatDataFrameBuf(data.tolist(), pack_pandas_index(prisoner.index), pack_pandas_index(prisoner.columns))

@laplace_mechanism_impl.register
def _(prisoner : SensitiveNDArray,
      *,
      eps      : float | None = None,
      scale    : float | None = None,
      ) -> FloatArrayBuf:
    sensitivity = float(prisoner.distance.max())
    assert_sensitivity(sensitivity)

    resolved_eps, resolved_scale = resolve_laplace_params(sensitivity, eps=eps, scale=scale)
    samples = _np.random.laplace(loc=prisoner._value, scale=resolved_scale)

    if isinstance(prisoner.accountant, PureAccountant):
        prisoner.accountant.spend(resolved_eps)
    elif isinstance(prisoner.accountant, ApproxAccountant):
        prisoner.accountant.spend((resolved_eps, 0.0))
    else:
        raise RuntimeError

    array = _np.asarray(samples)
    return FloatArrayBuf(values=array.reshape(-1).tolist(), shape=array.shape)

@overload
def gaussian_mechanism(prisoner : SensitiveInt | SensitiveFloat,
                       *,
                       eps      : floating | None = ...,
                       delta    : floating | None = ...,
                       rho      : floating | None = ...,
                       scale    : floating | None = ...,
                       ) -> float: ...

@overload
def gaussian_mechanism(prisoner : SensitiveSeries[Any],
                       *,
                       eps      : floating | None = ...,
                       delta    : floating | None = ...,
                       rho      : floating | None = ...,
                       scale    : floating | None = ...,
                       ) -> _pd.Series: ... # type: ignore[type-arg]

@overload
def gaussian_mechanism(prisoner : SensitiveDataFrame,
                       *,
                       eps      : floating | None = ...,
                       delta    : floating | None = ...,
                       rho      : floating | None = ...,
                       scale    : floating | None = ...,
                       ) -> _pd.DataFrame: ...

@overload
def gaussian_mechanism(prisoner : SensitiveNDArray,
                       *,
                       eps      : floating | None = ...,
                       delta    : floating | None = ...,
                       rho      : floating | None = ...,
                       scale    : floating | None = ...,
                       ) -> _npt.NDArray[Any]: ...

def gaussian_mechanism(prisoner : Any,
                       *,
                       eps      : floating | None = None,
                       delta    : floating | None = None,
                       rho      : floating | None = None,
                       scale    : floating | None = None,
                       ) -> float | _pd.Series | _pd.DataFrame | _npt.NDArray[Any]: # type: ignore[type-arg]
    result = gaussian_mechanism_impl(prisoner,
                                     eps   = float(eps)   if eps   is not None else None,
                                     delta = float(delta) if delta is not None else None,
                                     rho   = float(rho)   if rho   is not None else None,
                                     scale = float(scale) if scale is not None else None)

    if isinstance(result, float):
        return result
    if isinstance(result, FloatSeriesBuf):
        return _pd.Series(result.values, index=result.index.to_pandas(), name=result.name)
    elif isinstance(result, FloatDataFrameBuf):
        return _pd.DataFrame(result.values, index=result.index.to_pandas(), columns=result.columns.to_pandas())
    elif isinstance(result, FloatArrayBuf):
        return _np.asarray(result.values).reshape(result.shape)
    else:
        raise Exception

@egrpc.multifunction
def gaussian_mechanism_impl(prisoner : SensitiveInt | SensitiveFloat,
                            *,
                            eps      : float | None = None,
                            delta    : float | None = None,
                            rho      : float | None = None,
                            scale    : float | None = None,
                            ) -> float:
    sensitivity = float(prisoner.distance.max())
    assert_sensitivity(sensitivity)

    budget : BudgetType

    if isinstance(prisoner.accountant, PureAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxAccountant):
        resolved_eps, resolved_scale = resolve_gaussian_params_approx(sensitivity, eps=eps, delta=delta, scale=scale)
        assert delta is not None
        budget = (resolved_eps, delta)

    elif isinstance(prisoner.accountant, zCDPAccountant):
        resolved_rho, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho, scale=scale)
        budget = resolved_rho

    else:
        raise RuntimeError

    result = float(_np.random.normal(loc=prisoner._value, scale=resolved_scale))

    prisoner.accountant.spend(budget)

    return result

@gaussian_mechanism_impl.register
def _(prisoner : SensitiveSeries[realnum],
      *,
      eps      : float | None = None,
      delta    : float | None = None,
      rho      : float | None = None,
      scale    : float | None = None,
      ) -> FloatSeriesBuf:
    budget : BudgetType

    if isinstance(prisoner.accountant, PureAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxAccountant):
        assert delta is not None

        if prisoner._distance_group_axes == (0,):
            assert isinstance(prisoner._partitioned_distances, list)
            delta_each = delta / len(prisoner)

            if eps is not None:
                assert_eps(eps)
                eps_each = eps / len(prisoner)
                scales = []
                for distance in prisoner._partitioned_distances:
                    sensitivity = float(distance.max())
                    assert_sensitivity(sensitivity)
                    _, resolved_scale = resolve_gaussian_params_approx(sensitivity, eps=eps_each, delta=delta_each)
                    scales.append(resolved_scale)
                data = _np.random.normal(loc=prisoner._value, scale=scales)
                spent_eps = eps

            elif scale is not None:
                assert_gaussian_scale(scale)
                total_eps = 0.0
                for distance in prisoner._partitioned_distances:
                    sensitivity = float(distance.max())
                    assert_sensitivity(sensitivity)
                    resolved_eps, _ = resolve_gaussian_params_approx(sensitivity, delta=delta_each, scale=scale)
                    total_eps += resolved_eps
                data = _np.random.normal(loc=prisoner._value, scale=scale)
                spent_eps = total_eps

            else:
                raise Exception

            budget = (spent_eps, delta)

        else:
            sensitivity = float(prisoner.distance.max())
            resolved_eps, resolved_scale = resolve_gaussian_params_approx(sensitivity, eps=eps, delta=delta, scale=scale)
            data = _np.random.normal(loc=prisoner._value, scale=resolved_scale)
            budget = (resolved_eps, delta)

    elif isinstance(prisoner.accountant, zCDPAccountant):
        if prisoner._distance_group_axes == (0,):
            assert isinstance(prisoner._partitioned_distances, list)

            if rho is not None:
                assert_rho(rho)
                rho_each = rho / len(prisoner)
                scales = []
                for distance in prisoner._partitioned_distances:
                    sensitivity = float(distance.max())
                    assert_sensitivity(sensitivity)
                    _, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho_each)
                    scales.append(resolved_scale)
                data = _np.random.normal(loc=prisoner._value, scale=scales)
                spent_rho = rho
            else:
                assert scale is not None
                assert_gaussian_scale(scale)
                total_rho = 0.0
                for distance in prisoner._partitioned_distances:
                    sensitivity = float(distance.max())
                    assert_sensitivity(sensitivity)
                    resolved_rho, _ = resolve_gaussian_params_zcdp(sensitivity, scale=scale)
                    total_rho += resolved_rho
                data = _np.random.normal(loc=prisoner._value, scale=scale)
                spent_rho = total_rho

            budget = spent_rho

        else:
            sensitivity = float(prisoner.distance.max())
            resolved_rho, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho, scale=scale)
            data = _np.random.normal(loc=prisoner._value, scale=resolved_scale)
            budget = resolved_rho

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
      scale    : float | None = None,
      ) -> FloatDataFrameBuf:
    budget : BudgetType

    if isinstance(prisoner.accountant, PureAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxAccountant):
        assert delta is not None

        if prisoner._distance_group_axes == (1,):
            assert isinstance(prisoner._partitioned_distances, list)
            delta_each = delta / len(prisoner)

            if eps is not None:
                assert_eps(eps)
                eps_each = eps / len(prisoner)
                scales = []
                for distance in prisoner._partitioned_distances:
                    sensitivity = float(distance.max())
                    assert_sensitivity(sensitivity)
                    _, resolved_scale = resolve_gaussian_params_approx(sensitivity, eps=eps_each, delta=delta_each)
                    scales.append(resolved_scale)
                data = _np.random.normal(loc=prisoner._value, scale=scales)
                spent_eps = eps

            elif scale is not None:
                assert_gaussian_scale(scale)
                total_eps = 0.0
                for distance in prisoner._partitioned_distances:
                    sensitivity = float(distance.max())
                    assert_sensitivity(sensitivity)
                    resolved_eps, _ = resolve_gaussian_params_approx(sensitivity, delta=delta_each, scale=scale)
                    total_eps += resolved_eps
                data = _np.random.normal(loc=prisoner._value, scale=scale)
                spent_eps = total_eps

            else:
                raise Exception

            budget = (spent_eps, delta)

        else:
            sensitivity = float(prisoner.distance.max())
            resolved_eps, resolved_scale = resolve_gaussian_params_approx(sensitivity, eps=eps, delta=delta, scale=scale)
            data = _np.random.normal(loc=prisoner._value, scale=resolved_scale)
            budget = (resolved_eps, delta)

    elif isinstance(prisoner.accountant, zCDPAccountant):
        if prisoner._distance_group_axes == (1,):
            assert isinstance(prisoner._partitioned_distances, list)

            if rho is not None:
                assert_rho(rho)
                rho_each = rho / len(prisoner)
                scales = []
                for distance in prisoner._partitioned_distances:
                    sensitivity = float(distance.max())
                    assert_sensitivity(sensitivity)
                    _, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho_each)
                    scales.append(resolved_scale)
                data = _np.random.normal(loc=prisoner._value, scale=scales)
                spent_rho = rho

            elif scale is not None:
                assert_gaussian_scale(scale)
                total_rho = 0.0
                for distance in prisoner._partitioned_distances:
                    sensitivity = float(distance.max())
                    assert_sensitivity(sensitivity)
                    resolved_rho, _ = resolve_gaussian_params_zcdp(sensitivity, scale=scale)
                    total_rho += resolved_rho
                data = _np.random.normal(loc=prisoner._value, scale=scale)
                spent_rho = total_rho

            else:
                raise Exception

            budget = spent_rho

        else:
            sensitivity = float(prisoner.distance.max())
            resolved_rho, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho, scale=scale)
            data = _np.random.normal(loc=prisoner._value, scale=resolved_scale)
            budget = resolved_rho

    else:
        raise RuntimeError

    prisoner.accountant.spend(budget)

    return FloatDataFrameBuf(data.tolist(), pack_pandas_index(prisoner.index), pack_pandas_index(prisoner.columns))

@gaussian_mechanism_impl.register
def _(prisoner : SensitiveNDArray,
      *,
      eps      : float | None = None,
      delta    : float | None = None,
      rho      : float | None = None,
      scale    : float | None = None,
      ) -> FloatArrayBuf:
    if prisoner.norm_type != "l2":
        raise DPError("Gaussian mechanism on SensitiveNDArray requires L2 sensitivity. Use clip_norm(ord=None or 2).")

    sensitivity = float(prisoner.distance.max())
    assert_sensitivity(sensitivity)

    budget: BudgetType

    if isinstance(prisoner.accountant, PureAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxAccountant):
        resolved_eps, resolved_scale = resolve_gaussian_params_approx(sensitivity, eps=eps, delta=delta, scale=scale)
        assert delta is not None
        budget = (resolved_eps, delta)

    elif isinstance(prisoner.accountant, zCDPAccountant):
        resolved_rho, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho, scale=scale)
        budget = resolved_rho

    else:
        raise RuntimeError

    samples = _np.random.normal(loc=prisoner._value, scale=resolved_scale)

    prisoner.accountant.spend(budget)

    array = _np.asarray(samples)
    return FloatArrayBuf(values=array.reshape(-1).tolist(), shape=array.shape)

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

    budget : BudgetType

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
