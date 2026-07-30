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

from typing import Any

import egrpc
import numpy as _np
import numpy.typing as _npt

from ..accountants import BudgetType, PureDPAccountant, ApproxDPAccountant, zCDPAccountant, RDPAccountant, RDPSubsamplingAccountant
from ..mechanism import assert_sensitivity, resolve_gaussian_params_approx, resolve_gaussian_params_rdp, resolve_gaussian_params_zcdp, resolve_laplace_params
from ..util import DPError
from .array import SensitiveNDArray


@egrpc.function
def laplace_mechanism(prisoner : SensitiveNDArray,
      *,
      eps      : float | None = None,
      scale    : float | None = None,
      ) -> _npt.NDArray[Any]:
    sensitivity = float(prisoner._distance.max())
    assert_sensitivity(sensitivity)

    resolved_eps, resolved_scale = resolve_laplace_params(sensitivity, eps=eps, scale=scale)
    samples = _np.random.laplace(loc=prisoner._value, scale=resolved_scale)

    if isinstance(prisoner.accountant, PureDPAccountant):
        prisoner.accountant.spend(resolved_eps)
    elif isinstance(prisoner.accountant, ApproxDPAccountant):
        prisoner.accountant.spend((resolved_eps, 0.0))
    else:
        raise RuntimeError

    return _np.asarray(samples)

@egrpc.function
def gaussian_mechanism(prisoner : SensitiveNDArray,
      *,
      eps      : float | None = None,
      delta    : float | None = None,
      rho      : float | None = None,
      scale    : float | None = None,
      ) -> _npt.NDArray[Any]:
    sensitivity = float(prisoner._distance.max())
    assert_sensitivity(sensitivity)

    budget: BudgetType

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

    samples = _np.random.normal(loc=prisoner._value, scale=resolved_scale)

    prisoner.accountant.spend(budget)  # type: ignore[arg-type]

    return _np.asarray(samples)
