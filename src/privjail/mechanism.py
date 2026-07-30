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
import egrpc

from .util import DPError, floating, realnum
from .accountants import Accountant, BudgetType, PureDPAccountant, ApproxDPAccountant, zCDPAccountant, RDPAccountant, RDPBudgetType, RDPSubsamplingAccountant
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat
from .numpy import SensitiveNDArray
from .pandas import SensitiveSeries, SensitiveDataFrame

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

def rdp_eps_from_sigma(sensitivity: float, sigma: float, alpha: float) -> float:
    # Rényi Differential Privacy
    # https://arxiv.org/pdf/1702.07476
    # Example 1: Gaussian mechanism satisfies (α, αΔ²/(2σ²))-RDP
    return alpha * sensitivity * sensitivity / (2.0 * sigma * sigma)

def subsampled_rdp_eps_from_sigma(sensitivity   : float,
                                  sigma         : float,
                                  alpha         : float,
                                  sampling_rate : float,
                                  ) -> float:
    if sampling_rate <= 0 or sampling_rate > 1:
        raise ValueError("sampling_rate must be in (0, 1]")

    if not float(alpha).is_integer() or alpha < 2:
        raise ValueError("alpha must be an integer >= 2 for subsampled RDP")

    if sampling_rate == 1.0:
        return rdp_eps_from_sigma(sensitivity, sigma, alpha)

    # Rényi Differential Privacy of the Sampled Gaussian Mechanism
    # https://arxiv.org/abs/1908.10530
    # Theorem 9: A_α = Σ_{k=0}^{α} C(α,k) (1-q)^{α-k} q^k exp((k²-k) Δ²/(2σ²))
    # ε(α) = log(A_α) / (α-1)
    alpha_int = int(alpha)
    q = sampling_rate
    c = (sensitivity ** 2) / (2.0 * sigma ** 2)

    k = _np.arange(alpha_int + 1, dtype=float)
    log_binom = _np.array([math.lgamma(alpha_int + 1) - math.lgamma(ki + 1) - math.lgamma(alpha_int - ki + 1) for ki in k])
    log_terms = log_binom + (alpha_int - k) * _np.log(1 - q) + k * _np.log(q) + (k * k - k) * c
    log_a = float(_np.logaddexp.reduce(log_terms))

    return log_a / (alpha - 1)

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

def resolve_gaussian_params_rdp(sensitivity   : float,
                                alpha         : list[float],
                                *,
                                scale         : float | None = None,
                                sampling_rate : float | None = None,
                                ) -> tuple[RDPBudgetType, float]:
    if scale is None:
        raise ValueError("scale must be specified when using the Gaussian mechanism under RDP.")

    assert_gaussian_scale(scale)

    if sampling_rate is None:
        budget = {a: rdp_eps_from_sigma(sensitivity, scale, a) for a in alpha}
    else:
        budget = {a: subsampled_rdp_eps_from_sigma(sensitivity, scale, a, sampling_rate) for a in alpha}

    return budget, scale

def spend_gaussian(accountant  : Accountant[Any],
                   *,
                   sensitivity : float,
                   scale       : float,
                   delta       : float | None,
                   ) -> None:
    budget : BudgetType

    if isinstance(accountant, PureDPAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(accountant, ApproxDPAccountant):
        if delta is None:
            raise ValueError("delta must be specified for an ApproxDPAccountant.")
        epsilon, _ = resolve_gaussian_params_approx(sensitivity, delta=delta, scale=scale)
        budget = (epsilon, delta)

    elif isinstance(accountant, zCDPAccountant):
        budget, _ = resolve_gaussian_params_zcdp(sensitivity, scale=scale)

    elif isinstance(accountant, RDPAccountant):
        sampling_rate = accountant.parent.sampling_rate if isinstance(accountant.parent, RDPSubsamplingAccountant) else None
        budget, _ = resolve_gaussian_params_rdp(sensitivity,
                                                accountant.alpha,
                                                scale=scale,
                                                sampling_rate=sampling_rate)

    else:
        raise TypeError(f"Unsupported accountant for Gaussian release: {type(accountant).__name__}.")

    accountant.spend(budget)  # type: ignore[arg-type]

@overload
def laplace_mechanism(prisoner : SensitiveInt | SensitiveFloat,
                      *,
                      eps      : float | None = ...,
                      scale    : float | None = ...,
                      ) -> float: ...

@overload
def laplace_mechanism(prisoner : SensitiveSeries[Any],
                      *,
                      eps      : float | None = ...,
                      scale    : float | None = ...,
                      ) -> _pd.Series: ...

@overload
def laplace_mechanism(prisoner : SensitiveDataFrame,
                      *,
                      eps      : float | None = ...,
                      scale    : float | None = ...,
                      ) -> _pd.DataFrame: ...

@overload
def laplace_mechanism(prisoner : SensitiveNDArray,
                      *,
                      eps      : float | None = ...,
                      scale    : float | None = ...,
                      ) -> _npt.NDArray[Any]: ...

def laplace_mechanism(prisoner : SensitiveInt | SensitiveFloat | SensitiveSeries[Any] | SensitiveDataFrame | SensitiveNDArray,
                      *,
                      eps      : float | None = None,
                      scale    : float | None = None,
                      ) -> Any:
    if isinstance(prisoner, (SensitiveInt, SensitiveFloat)):
        return _laplace_scalar(prisoner, eps=eps, scale=scale)
    if isinstance(prisoner, (SensitiveSeries, SensitiveDataFrame)):
        from .pandas.mechanism import laplace_mechanism as pandas_laplace
        return pandas_laplace(prisoner, eps=eps, scale=scale)
    if isinstance(prisoner, SensitiveNDArray):
        from .numpy.mechanism import laplace_mechanism as numpy_laplace
        return numpy_laplace(prisoner, eps=eps, scale=scale)
    raise TypeError(f"Unsupported Laplace mechanism input: {type(prisoner).__name__}.")

@egrpc.function
def _laplace_scalar(prisoner : SensitiveInt | SensitiveFloat,
                    *,
                    eps      : float | None = None,
                    scale    : float | None = None,
                    ) -> float:
    sensitivity = float(prisoner._distance.max())
    assert_sensitivity(sensitivity)

    resolved_eps, resolved_scale = resolve_laplace_params(sensitivity, eps=eps, scale=scale)

    result = float(_np.random.laplace(loc=prisoner._value, scale=resolved_scale))

    if isinstance(prisoner.accountant, PureDPAccountant):
        prisoner.accountant.spend(resolved_eps)
    elif isinstance(prisoner.accountant, ApproxDPAccountant):
        prisoner.accountant.spend((resolved_eps, 0.0))
    else:
        raise RuntimeError

    return result

@overload
def gaussian_mechanism(prisoner : SensitiveInt | SensitiveFloat,
                       *,
                       eps      : float | None = ...,
                       delta    : float | None = ...,
                       rho      : float | None = ...,
                       scale    : float | None = ...,
                       ) -> float: ...

@overload
def gaussian_mechanism(prisoner : SensitiveSeries[Any],
                       *,
                       eps      : float | None = ...,
                       delta    : float | None = ...,
                       rho      : float | None = ...,
                       scale    : float | None = ...,
                       ) -> _pd.Series: ...

@overload
def gaussian_mechanism(prisoner : SensitiveDataFrame,
                       *,
                       eps      : float | None = ...,
                       delta    : float | None = ...,
                       rho      : float | None = ...,
                       scale    : float | None = ...,
                       ) -> _pd.DataFrame: ...

@overload
def gaussian_mechanism(prisoner : SensitiveNDArray,
                       *,
                       eps      : float | None = ...,
                       delta    : float | None = ...,
                       rho      : float | None = ...,
                       scale    : float | None = ...,
                       ) -> _npt.NDArray[Any]: ...

@overload
def gaussian_mechanism(prisoner : Any,
                       *,
                       eps      : float | None = ...,
                       delta    : float | None = ...,
                       rho      : float | None = ...,
                       scale    : float | None = ...,
                       ) -> Any: ...

def gaussian_mechanism(prisoner : Any,
                       *,
                       eps      : float | None = None,
                       delta    : float | None = None,
                       rho      : float | None = None,
                       scale    : float | None = None,
                       ) -> Any:
    if isinstance(prisoner, (SensitiveInt, SensitiveFloat)):
        return _gaussian_scalar(prisoner, eps=eps, delta=delta, rho=rho, scale=scale)
    if isinstance(prisoner, (SensitiveSeries, SensitiveDataFrame)):
        from .pandas.mechanism import gaussian_mechanism as pandas_gaussian
        return pandas_gaussian(prisoner, eps=eps, delta=delta, rho=rho, scale=scale)
    if isinstance(prisoner, SensitiveNDArray):
        from .numpy.mechanism import gaussian_mechanism as numpy_gaussian
        return numpy_gaussian(prisoner, eps=eps, delta=delta, rho=rho, scale=scale)

    if eps is not None or rho is not None:
        raise ValueError("JAX Gaussian release currently accepts scale and delta.")
    if scale is None:
        raise ValueError("scale must be specified.")

    import jax
    import jax.numpy as jnp
    from .jax.array import SensitiveArray as JaxSensitiveArray
    from .jax.mechanism import gaussian_mechanism as jax_gaussian
    from .jax.util import require_static

    require_static(scale, "gaussian_mechanism scale")
    require_static(delta, "gaussian_mechanism delta")
    leaves, treedef = jax.tree.flatten(prisoner)
    if not leaves:
        raise ValueError("gaussian_mechanism requires at least one array.")
    arrays = [
        leaf if isinstance(leaf, JaxSensitiveArray) else jnp.asarray(leaf)
        for leaf in leaves
    ]
    noisy = jax_gaussian(arrays, scale, delta)
    return jax.tree.unflatten(treedef, noisy)

@egrpc.function
def _gaussian_scalar(prisoner : SensitiveInt | SensitiveFloat,
                     *,
                     eps      : float | None = None,
                     delta    : float | None = None,
                     rho      : float | None = None,
                     scale    : float | None = None,
                     ) -> float:
    sensitivity = float(prisoner._distance.max())
    assert_sensitivity(sensitivity)

    budget : BudgetType

    if isinstance(prisoner.accountant, PureDPAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxDPAccountant):
        resolved_eps, resolved_scale = resolve_gaussian_params_approx(sensitivity, eps=eps, delta=delta, scale=scale)
        assert delta is not None
        budget = (resolved_eps, delta)

    elif isinstance(prisoner.accountant, zCDPAccountant):
        resolved_rho, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho, scale=scale)
        budget = resolved_rho

    elif isinstance(prisoner.accountant, RDPAccountant):
        # Check if under subsampling context
        sampling_rate: float | None = None
        if isinstance(prisoner.accountant.parent, RDPSubsamplingAccountant):
            sampling_rate = prisoner.accountant.parent.sampling_rate

        rdp_budget, resolved_scale = resolve_gaussian_params_rdp(
            sensitivity, prisoner.accountant.alpha, scale=scale, sampling_rate=sampling_rate
        )
        budget = rdp_budget

    else:
        raise RuntimeError

    result = float(_np.random.normal(loc=prisoner._value, scale=resolved_scale))

    prisoner.accountant.spend(budget)  # type: ignore[arg-type]

    return result

@egrpc.function
def exponential_mechanism(scores : Sequence[SensitiveInt | SensitiveFloat],
                          *,
                          eps    : float | None = None,
                          rho    : float | None = None,
                          ) -> int:
    if len(scores) == 0:
        raise ValueError("scores must have at least one element.")

    sensitivity = max([v._distance.max() for v in scores])
    assert_sensitivity(sensitivity)

    # create a dummy prisoner to propagate budget consumption to all prisoners
    prisoner_dummy = Prisoner(0, scores[0]._distance, parents=scores)

    budget : BudgetType

    if isinstance(prisoner_dummy.accountant, PureDPAccountant):
        assert eps is not None
        assert_eps(eps)
        budget = eps

    elif isinstance(prisoner_dummy.accountant, ApproxDPAccountant):
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

    prisoner_dummy.accountant.spend(budget)  # type: ignore[arg-type]

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
