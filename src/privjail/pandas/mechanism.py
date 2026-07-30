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

import egrpc
import numpy as _np
import pandas as _pd

from ..accountants import BudgetType, PureDPAccountant, ApproxDPAccountant, zCDPAccountant, RDPAccountant, RDPBudgetType, RDPSubsamplingAccountant
from ..mechanism import assert_eps, assert_gaussian_scale, assert_laplace_scale, assert_rho, assert_sensitivity, resolve_gaussian_params_approx, resolve_gaussian_params_rdp, resolve_gaussian_params_zcdp, resolve_laplace_params
from ..util import DPError, realnum
from .dataframe import SensitiveDataFrame
from .series import SensitiveSeries


@egrpc.multifunction
def laplace_mechanism(prisoner : SensitiveSeries[realnum],
      *,
      eps      : float | None = None,
      scale    : float | None = None,
      ) -> _pd.Series:
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
        sensitivity = float(prisoner._distance.max())
        resolved_eps, resolved_scale = resolve_laplace_params(sensitivity, eps=eps, scale=scale)
        data = _np.random.laplace(loc=prisoner._value, scale=resolved_scale)
        spent_eps = resolved_eps

    if isinstance(prisoner.accountant, PureDPAccountant):
        prisoner.accountant.spend(spent_eps)
    elif isinstance(prisoner.accountant, ApproxDPAccountant):
        prisoner.accountant.spend((spent_eps, 0.0))
    else:
        raise RuntimeError

    return _pd.Series(data, index=prisoner.index, name=prisoner.name)

@laplace_mechanism.register
def _(prisoner : SensitiveDataFrame,
      *,
      eps      : float | None = None,
      scale    : float | None = None,
      ) -> _pd.DataFrame:
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
        sensitivity = float(prisoner._distance.max())
        resolved_eps, resolved_scale = resolve_laplace_params(sensitivity, eps=eps, scale=scale)
        data = _np.random.laplace(loc=prisoner._value, scale=resolved_scale)
        spent_eps = resolved_eps

    if isinstance(prisoner.accountant, PureDPAccountant):
        prisoner.accountant.spend(spent_eps)
    elif isinstance(prisoner.accountant, ApproxDPAccountant):
        prisoner.accountant.spend((spent_eps, 0.0))
    else:
        raise RuntimeError

    return _pd.DataFrame(data, index=prisoner.index, columns=prisoner.columns)

@egrpc.multifunction
def gaussian_mechanism(prisoner : SensitiveSeries[realnum],
      *,
      eps      : float | None = None,
      delta    : float | None = None,
      rho      : float | None = None,
      scale    : float | None = None,
      ) -> _pd.Series:
    budget : BudgetType

    if isinstance(prisoner.accountant, PureDPAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxDPAccountant):
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
            sensitivity = float(prisoner._distance.max())
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
            sensitivity = float(prisoner._distance.max())
            resolved_rho, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho, scale=scale)
            data = _np.random.normal(loc=prisoner._value, scale=resolved_scale)
            budget = resolved_rho

    elif isinstance(prisoner.accountant, RDPAccountant):
        alpha = prisoner.accountant.alpha

        # Check if under subsampling context
        sampling_rate: float | None = None
        if isinstance(prisoner.accountant.parent, RDPSubsamplingAccountant):
            sampling_rate = prisoner.accountant.parent.sampling_rate

        if prisoner._distance_group_axes == (0,):
            assert isinstance(prisoner._partitioned_distances, list)

            # Subsampling not supported with partitioned distances
            if sampling_rate is not None:
                raise DPError("Subsampled RDP is not supported with partitioned distances")

            assert scale is not None
            assert_gaussian_scale(scale)
            total_budget: RDPBudgetType = {a: 0.0 for a in alpha}
            for distance in prisoner._partitioned_distances:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                elem_budget, _ = resolve_gaussian_params_rdp(sensitivity, alpha, scale=scale)
                for a in alpha:
                    total_budget[a] += elem_budget[a]
            data = _np.random.normal(loc=prisoner._value, scale=scale)
            budget = total_budget

        else:
            sensitivity = float(prisoner._distance.max())
            rdp_budget, resolved_scale = resolve_gaussian_params_rdp(
                sensitivity, alpha, scale=scale, sampling_rate=sampling_rate
            )
            data = _np.random.normal(loc=prisoner._value, scale=resolved_scale)
            budget = rdp_budget

    else:
        raise RuntimeError

    prisoner.accountant.spend(budget)  # type: ignore[arg-type]

    return _pd.Series(data, index=prisoner.index, name=prisoner.name)

@gaussian_mechanism.register
def _(prisoner : SensitiveDataFrame,
      *,
      eps      : float | None = None,
      delta    : float | None = None,
      rho      : float | None = None,
      scale    : float | None = None,
      ) -> _pd.DataFrame:
    budget : BudgetType

    if isinstance(prisoner.accountant, PureDPAccountant):
        raise DPError("Gaussian mechanism cannot be used under Pure DP")

    elif isinstance(prisoner.accountant, ApproxDPAccountant):
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
            sensitivity = float(prisoner._distance.max())
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
            sensitivity = float(prisoner._distance.max())
            resolved_rho, resolved_scale = resolve_gaussian_params_zcdp(sensitivity, rho=rho, scale=scale)
            data = _np.random.normal(loc=prisoner._value, scale=resolved_scale)
            budget = resolved_rho

    elif isinstance(prisoner.accountant, RDPAccountant):
        alpha = prisoner.accountant.alpha

        # Check if under subsampling context
        sampling_rate: float | None = None
        if isinstance(prisoner.accountant.parent, RDPSubsamplingAccountant):
            sampling_rate = prisoner.accountant.parent.sampling_rate

        if prisoner._distance_group_axes == (1,):
            assert isinstance(prisoner._partitioned_distances, list)

            # Subsampling not supported with partitioned distances
            if sampling_rate is not None:
                raise DPError("Subsampled RDP is not supported with partitioned distances")

            assert scale is not None
            assert_gaussian_scale(scale)
            total_budget: RDPBudgetType = {a: 0.0 for a in alpha}
            for distance in prisoner._partitioned_distances:
                sensitivity = float(distance.max())
                assert_sensitivity(sensitivity)
                elem_budget, _ = resolve_gaussian_params_rdp(sensitivity, alpha, scale=scale)
                for a in alpha:
                    total_budget[a] += elem_budget[a]
            data = _np.random.normal(loc=prisoner._value, scale=scale)
            budget = total_budget

        else:
            sensitivity = float(prisoner._distance.max())
            rdp_budget, resolved_scale = resolve_gaussian_params_rdp(
                sensitivity, alpha, scale=scale, sampling_rate=sampling_rate
            )
            data = _np.random.normal(loc=prisoner._value, scale=resolved_scale)
            budget = rdp_budget

    else:
        raise RuntimeError

    prisoner.accountant.spend(budget)  # type: ignore[arg-type]

    return _pd.DataFrame(data, index=prisoner.index, columns=prisoner.columns)
