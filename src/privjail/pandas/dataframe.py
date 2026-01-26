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

from __future__ import annotations
from typing import TypeVar, Any, Sequence, Mapping, TYPE_CHECKING
import copy
import math

import numpy as _np
import pandas as _pd

from .. import egrpc
from ..util import DPError, ElementType, floating, is_floating, is_integer, is_realnum
from ..array_base import PrivArrayBase
from ..alignment import assert_distance_axis
from ..prisoner import Prisoner, SensitiveInt, SensitiveFloat
from ..realexpr import RealExpr
from ..accountants import Accountant
from ..numpy import PrivNDArray, SensitiveNDArray, NDArrayDomain
from .util import ColumnType, ColumnsType
from .domain import Domain, BoolDomain, RealDomain, sum_sensitivity
from .series import PrivSeries, SensitiveSeries

if TYPE_CHECKING:
    from .groupby import PrivDataFrameGroupBy, PrivDataFrameGroupByUser

T = TypeVar("T")

@egrpc.remoteclass
class PrivDataFrame(PrivArrayBase[_pd.DataFrame]):
    """Private DataFrame.

    Each row in this dataframe object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of rows is treated as a sensitive value.
    """
    _domains       : Mapping[str, Domain]
    _user_key      : str | None
    _user_max_freq : RealExpr

    def __init__(self,
                 data                   : Any,
                 domains                : Mapping[str, Domain],
                 distance               : RealExpr,
                 user_key               : str | None                   = None,
                 user_max_freq          : RealExpr                     = RealExpr.INF,
                 index                  : Any                          = None,
                 columns                : Any                          = None,
                 dtype                  : Any                          = None,
                 copy                   : bool                         = False,
                 *,
                 parents                : Sequence[PrivArrayBase[Any]] = [],
                 accountant             : Accountant[Any] | None       = None,
                 inherit_axis_signature : bool                         = False,
                 ):
        self._domains = domains
        self._user_key = user_key
        self._user_max_freq = user_max_freq
        df = _pd.DataFrame(data, index, columns, dtype, copy)
        assert user_key is None or user_key in df.columns
        super().__init__(value                  = df,
                         distance               = distance,
                         distance_axis          = 0,
                         parents                = parents,
                         accountant             = accountant,
                         inherit_axis_signature = inherit_axis_signature)

    def _is_eldp(self) -> bool:
        return self._user_key is None

    def _is_uldp(self) -> bool:
        return self._user_key is not None

    def _assert_not_uldp(self) -> None:
        if self._is_uldp():
            raise DPError("This operation is not permitted for user DataFrame.")

    def _assert_user_key_not_in(self, columns: Sequence[str]) -> None:
        if self._user_key is not None and self._user_key in columns:
            raise DPError("This operation is not permitted for the user key of user DataFrame.")

    def _eldp_distance(self) -> RealExpr:
        if self._is_eldp():
            return self.distance
        else:
            return self.distance * self._user_max_freq

    def _get_dummy_df(self, n_rows: int = 3) -> _pd.DataFrame:
        index = list(range(n_rows)) + ['...']
        columns = self.columns
        dummy_data = [['***' for _ in columns] for _ in range(n_rows)] + [['...' for _ in columns]]
        return _pd.DataFrame(dummy_data, index=index, columns=columns)

    def __str__(self) -> str:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_df().__str__()

    def __repr__(self) -> str:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_df().__repr__()

    def _repr_html_(self) -> Any:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_df()._repr_html_() # type: ignore

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(df) is not supported. Use df.shape[0] instead.")

    @egrpc.multimethod
    def __getitem__(self, key: ColumnType) -> PrivSeries[ElementType]:
        # TODO: consider duplicated column names
        value_type = self.domains[key].type()
        user_key_included = self._user_key == key
        # TODO: how to pass `value_type` from server to client via egrpc?
        return PrivSeries[value_type](data                   = self._value.__getitem__(key), # type: ignore[valid-type]
                                      domain                 = self.domains[key],
                                      distance               = self.distance if user_key_included else self._eldp_distance(),
                                      is_user_key            = user_key_included,
                                      user_max_freq          = self._user_max_freq if user_key_included else RealExpr.INF,
                                      parents                = [self],
                                      inherit_axis_signature = True)

    @__getitem__.register
    def _(self, key: ColumnsType) -> PrivDataFrame:
        new_domains = {c: self.domains[c] for c in key}
        user_key_included = self._user_key in key
        return PrivDataFrame(data                   = self._value.__getitem__(key),
                             domains                = new_domains,
                             distance               = self.distance if user_key_included else self._eldp_distance(),
                             user_key               = self._user_key if user_key_included else None,
                             user_max_freq          = self._user_max_freq if user_key_included else RealExpr.INF,
                             parents                = [self],
                             inherit_axis_signature = True)

    @__getitem__.register
    def _(self, key: PrivSeries[bool]) -> PrivDataFrame:
        assert_distance_axis(self, key)
        return PrivDataFrame(data                   = self._value.__getitem__(key._value),
                             domains                = self.domains,
                             distance               = self.distance,
                             user_key               = self._user_key,
                             user_max_freq          = self._user_max_freq,
                             parents                = [self, key],
                             inherit_axis_signature = False)

    @egrpc.multimethod
    def __setitem__(self, key: ColumnType, value: ElementType) -> None:
        self._assert_user_key_not_in([key])
        # TODO: consider domain transform
        self._value[key] = value

    @__setitem__.register
    def _(self, key: ColumnType, value: PrivSeries[Any]) -> None:
        self._domains = dict(self.domains) | {key: value.domain}

        if value._is_uldp():
            # Even if the original df has a user key, overwrite the key with the new one
            self._user_key = key
            self._user_max_freq = value._user_max_freq
        elif self._user_key == key:
            # This df is no longer a user df
            self._user_key = None
            self._user_max_freq = RealExpr.INF

        self._value[key] = value._value
        # TODO: add `value` to parents?

    @__setitem__.register
    def _(self, key: ColumnType, value: PrivNDArray) -> None:
        assert_distance_axis(self, value)

        if value.ndim != 1:
            raise ValueError("Only 1D arrays can be assigned as a column.")

        bound = value.domain.norm_bound
        if bound is None:
            new_domain = RealDomain(dtype="float64", range=(None, None))
        else:
            new_domain = RealDomain(dtype="float64", range=(-bound, bound))

        self._domains = dict(self.domains) | {key: new_domain}

        if self._user_key == key:
            # This df is no longer a user df
            self._user_key = None
            self._user_max_freq = RealExpr.INF

        self._value[key] = value._value

    @__setitem__.register
    def _(self, key: ColumnsType, value: ElementType) -> None:
        # TODO: consider domain transform
        self._assert_user_key_not_in(key)
        self._value[key] = value

    @__setitem__.register
    def _(self, key: ColumnsType, value: PrivDataFrame) -> None:
        # TODO: consider domain transform
        self._assert_user_key_not_in(key)
        value._assert_not_uldp()
        self._value[key] = value._value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: ElementType) -> None:
        # TODO: consider domain transform
        assert_distance_axis(self, key)
        self._assert_not_uldp()
        self._value[key._value] = value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: Sequence[ElementType]) -> None:
        # TODO: consider domain transform
        assert_distance_axis(self, key)
        self._assert_not_uldp()
        self._value[key._value] = value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: PrivDataFrame) -> None:
        # TODO: consider domain transform
        assert_distance_axis(self, key)
        self._assert_not_uldp()
        value._assert_not_uldp()
        self._value[key._value] = value._value

    @egrpc.multimethod
    def __eq__(self, other: PrivDataFrame) -> PrivDataFrame:
        assert_distance_axis(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data                   = self._value == other._value,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self, other],
                             inherit_axis_signature = True)

    @__eq__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data                   = self._value == other,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self],
                             inherit_axis_signature = True)

    @egrpc.multimethod
    def __ne__(self, other: PrivDataFrame) -> PrivDataFrame:
        assert_distance_axis(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data                   = self._value != other._value,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self, other],
                             inherit_axis_signature = True)

    @__ne__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data                   = self._value != other,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self],
                             inherit_axis_signature = True)

    @egrpc.multimethod
    def __lt__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        assert_distance_axis(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data                   = self._value < other._value,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self, other],
                             inherit_axis_signature = True)

    @__lt__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data                   = self._value < other,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self],
                             inherit_axis_signature = True)

    @egrpc.multimethod
    def __le__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        assert_distance_axis(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data                   = self._value <= other._value,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self, other],
                             inherit_axis_signature = True)

    @__le__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data                   = self._value <= other,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self],
                             inherit_axis_signature = True)

    @egrpc.multimethod
    def __gt__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        assert_distance_axis(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data                   = self._value > other._value,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self, other],
                             inherit_axis_signature = True)

    @__gt__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data                   = self._value > other,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self],
                             inherit_axis_signature = True)

    @egrpc.multimethod
    def __ge__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        assert_distance_axis(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data                   = self._value >= other._value,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self, other],
                             inherit_axis_signature = True)

    @__ge__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data                   = self._value >= other,
                             domains                = {c: BoolDomain() for c in self.domains},
                             distance               = self.distance,
                             parents                = [self],
                             inherit_axis_signature = True)

    @egrpc.property
    def shape(self) -> tuple[SensitiveInt, int]:
        nrows = SensitiveInt(value=self._value.shape[0], distance=self._eldp_distance(), parents=[self])
        ncols = self._value.shape[1]
        return (nrows, ncols)

    @egrpc.property
    def size(self) -> SensitiveInt:
        return SensitiveInt(value=self._value.size, distance=self._eldp_distance() * len(self._value.columns), parents=[self])

    # TODO: define privjail's own Index[T] type
    @egrpc.property
    def columns(self) -> _pd.Index: # type: ignore
        return self._value.columns

    # FIXME
    @property
    def dtypes(self) -> _pd.Series[Any]:
        return self._value.dtypes

    @egrpc.property
    def domains(self) -> Mapping[ColumnType, Domain]:
        return self._domains

    @egrpc.property
    def user_key(self) -> ColumnType | None:
        return self._user_key

    @egrpc.property
    def user_max_freq(self) -> int | None:
        return int(self._user_max_freq.max()) if not self._user_max_freq.is_inf() else None

    @egrpc.method
    def to_numpy(self, copy: bool | None = None) -> PrivNDArray:
        self._assert_not_uldp()

        if copy is None or not copy:
            raise NotImplementedError("`copy` must be True in privjail to_numpy().")

        if not all(isinstance(domain, RealDomain) for domain in self.domains.values()):
            raise DPError("All columns must have RealDomain to be converted to NDArray.")

        array = self._value.to_numpy(dtype=float, copy=copy)

        l1_norm_max = 0.0
        for domain in self.domains.values():
            assert isinstance(domain, RealDomain)
            lower, upper = domain.range
            lower_abs = _np.abs(lower) if lower is not None else math.inf
            upper_abs = _np.abs(upper) if upper is not None else math.inf
            l1_norm_max += max(lower_abs, upper_abs)

        new_domain = NDArrayDomain(norm_type="l1", norm_bound=l1_norm_max if l1_norm_max != math.inf else None)
        return PrivNDArray(value                  = array,
                           distance               = self.distance,
                           distance_axis          = 0,
                           domain                 = new_domain,
                           parents                = [self],
                           inherit_axis_signature = True)

    @property
    def values(self) -> PrivNDArray:
        return self.to_numpy(copy=True)

    # TODO: add test
    @egrpc.method
    def head(self, n: int = 5) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data                   = self._value.head(n),
                             domains                = self.domains,
                             distance               = self.distance * 2,
                             parents                = [self],
                             inherit_axis_signature = False)

    # TODO: add test
    @egrpc.method
    def tail(self, n: int = 5) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data                   = self._value.tail(n),
                             domains                = self.domains,
                             distance               = self.distance * 2,
                             parents                = [self],
                             inherit_axis_signature = False)

    @egrpc.method
    def drop(self,
             labels  : ColumnType | ColumnsType | None = None,
             *,
             axis    : int | str                       = 0, # 0, 1, "index", "columns"
             index   : ColumnType | ColumnsType | None = None,
             columns : ColumnType | ColumnsType | None = None,
             level   : int | None                      = None,
             errors  : str                             = "raise", # "raise" | "ignore"
             ) -> PrivDataFrame:
        if axis not in (1, "columns") or index is not None:
            raise DPError("Rows cannot be dropped")

        if isinstance(labels, ColumnType):
            drop_columns = [labels]
        elif isinstance(labels, Sequence):
            drop_columns = list(labels)
        else:
            raise TypeError

        new_domains = {k: v for k, v in self.domains.items() if k not in drop_columns}
        user_key_included = self._is_uldp() and self._user_key not in drop_columns

        df = self._value.drop(labels, axis=axis, index=index, columns=columns, level=level, errors=errors) # type: ignore
        return PrivDataFrame(data                   = df,
                             domains                = new_domains,
                             distance               = self.distance if user_key_included else self._eldp_distance(),
                             user_key               = self._user_key if user_key_included else None,
                             user_max_freq          = self._user_max_freq if user_key_included else RealExpr.INF,
                             parents                = [self],
                             inherit_axis_signature = True)

    # TODO: add test
    @egrpc.method
    def sort_values(self,
                    by        : ColumnType | ColumnsType,
                    *,
                    ascending : bool = True,
                    ) -> PrivDataFrame:
        df = self._value.sort_values(by, ascending=ascending, kind="stable")
        return PrivDataFrame(data                   = df,
                             domains                = self.domains,
                             distance               = self.distance,
                             user_key               = self._user_key,
                             user_max_freq          = self._user_max_freq,
                             parents                = [self],
                             inherit_axis_signature = False)

    @egrpc.method
    def replace(self,
                to_replace : ElementType | None = None,
                value      : ElementType | None = None,
                ) -> PrivDataFrame:
        self._assert_not_uldp()

        if (not is_realnum(to_replace)) or (not is_realnum(value)):
            # TODO: consider string and category dtype
            raise NotImplementedError

        new_domains = dict()
        for col, domain in self.domains.items():
            if domain.dtype == "int64" and _np.isnan(value):
                new_domain = copy.copy(domain)
                new_domain.dtype = "Int64"
                new_domains[col] = new_domain

            elif isinstance(domain, RealDomain):
                a, b = domain.range
                if (a is None or a <= to_replace) and (b is None or to_replace <= b):
                    new_a = min(a, value) if a is not None else None # type: ignore[type-var]
                    new_b = max(b, value) if b is not None else None # type: ignore[type-var]

                    new_domain = copy.copy(domain)
                    new_domain.range = (new_a, new_b)
                    new_domains[col] = new_domain

                else:
                    new_domains[col] = domain

            else:
                new_domains[col] = domain

        df = self._value.replace(to_replace, value) # type: ignore[arg-type]
        return PrivDataFrame(data                   = df,
                             domains                = new_domains,
                             distance               = self.distance,
                             parents                = [self],
                             inherit_axis_signature = True)

    @egrpc.method
    def dropna(self, *, ignore_index: bool = False) -> PrivDataFrame:
        if ignore_index:
            raise DPError("`ignore_index` must be False. Index cannot be reindexed with positions.")

        new_domains = dict()
        for col, domain in self.domains.items():
            if domain.dtype == "Int64":
                new_domain = copy.copy(domain)
                new_domain.dtype = "int64"
                new_domains[col] = new_domain
            else:
                new_domains[col] = domain

        return PrivDataFrame(data                   = self._value.dropna(),
                             domains                = new_domains,
                             distance               = self.distance,
                             user_key               = self._user_key,
                             user_max_freq          = self._user_max_freq,
                             parents                = [self],
                             inherit_axis_signature = True)

    def groupby(self,
                by         : ColumnType | ColumnsType,
                level      : int | None = None, # TODO: support multiindex?
                as_index   : bool       = True,
                sort       : bool       = True,
                group_keys : bool       = True,
                observed   : bool       = True,
                dropna     : bool       = True,
                ) -> PrivDataFrameGroupBy | PrivDataFrameGroupByUser:
        from .groupby import _do_group_by
        return _do_group_by(self, by, level=level, as_index=as_index, sort=sort,
                            group_keys=group_keys, observed=observed, dropna=dropna)

    @egrpc.method
    def sum(self) -> SensitiveSeries[int] | SensitiveSeries[float]:
        distances = [self.distance * sum_sensitivity(domain) for col, domain in self.domains.items()]
        data = self._value.sum()
        if all(domain.dtype in ("int64", "Int64") for domain in self.domains.values()):
            return SensitiveSeries[int](data,
                                        distance_group_axes   = (0,),
                                        partitioned_distances = distances,
                                        parents               = [self])
        else:
            return SensitiveSeries[float](data,
                                          distance_group_axes   = (0,),
                                          partitioned_distances = distances,
                                          parents               = [self])

    def mean(self, eps: float) -> _pd.Series[float]:
        eps_each = eps / len(self.columns)
        data = [self[col].mean(eps=eps_each) for col in self.columns]
        return _pd.Series(data, index=self.columns) # type: ignore[no-any-return]

    @egrpc.method
    def sample(self,
               n       : int | None = None,
               frac    : int | None = None,
               replace : bool       = False,
               ) -> PrivDataFrame:
        # TODO: consider more parameters
        assert n is None
        assert frac is not None
        assert frac <= 1
        assert not replace
        return PrivDataFrame(data                   = self._value.sample(frac=frac),
                             domains                = self.domains,
                             distance               = self.distance,
                             user_key               = self._user_key,
                             user_max_freq          = self._user_max_freq,
                             parents                = [self],
                             inherit_axis_signature = False)

@egrpc.remoteclass
class SensitiveDataFrame(Prisoner[_pd.DataFrame]):
    _distance_group_axes   : tuple[int, ...] | None
    _partitioned_distances : list[RealExpr] | None
    _distributed_df        : _pd.DataFrame | None

    def __init__(self,
                 data                  : Any,
                 distance_group_axes   : tuple[int, ...] | None    = None,
                 distance              : RealExpr | None           = None,
                 partitioned_distances : Sequence[RealExpr] | None = None,
                 index                 : Any                       = None,
                 columns               : Any                       = None,
                 *,
                 parents               : Sequence[Prisoner[Any]]   = [],
                 accountant            : Accountant[Any] | None    = None,
                 distributed_df        : _pd.DataFrame | None      = None,
                 ):
        self._distance_group_axes = tuple(distance_group_axes) if distance_group_axes is not None else None
        self._partitioned_distances = list(partitioned_distances) if partitioned_distances is not None else None
        self._distributed_df = distributed_df

        df = _pd.DataFrame(data, index=index, columns=columns)

        if self._distance_group_axes is None:
            if distance is None:
                raise ValueError("`distance` must be specified when distance_group_axes is None.")
            if self._partitioned_distances is not None:
                raise ValueError("`partitioned_distances` must not be provided when distance_group_axes is None.")

        elif self._distance_group_axes == (1,):
            if self._partitioned_distances is None:
                raise ValueError("`partitioned_distances` is required when distance_group_axes=(1,).")
            if len(df.columns) != len(self._partitioned_distances):
                raise ValueError("`partitioned_distances` length must match the number of columns.")

            distance = sum(self._partitioned_distances, start=RealExpr(0))

        else:
            raise ValueError("Unsupported distance_group_axes for SensitiveDataFrame.")

        super().__init__(value=df, distance=distance, parents=parents, accountant=accountant)

    def _distance_of(self, key: ElementType) -> RealExpr:
        distances = self._ensure_partitioned_distances()
        icol = self.columns.get_loc(key)
        assert isinstance(icol, int)
        return distances[icol]

    def _ensure_partitioned_distances(self) -> list[RealExpr]:
        if self._distance_group_axes is None:
            if self._partitioned_distances is None:
                self._partitioned_distances = self.distance.create_exclusive_children(len(self.columns))

        elif self._distance_group_axes == (1,):
            assert self._partitioned_distances is not None

        else:
            raise ValueError("Unsupported distance_group_axes for SensitiveDataFrame.")

        assert self._partitioned_distances is not None
        return self._partitioned_distances

    def _wrap_sensitive_value(self,
                              value      : ElementType,
                              column     : ElementType,
                              distance   : RealExpr,
                              parents    : Sequence[Prisoner[Any]],
                              accountant : Accountant[Any] | None = None,
                              ) -> SensitiveInt | SensitiveFloat:
        dtype = self.dtypes[column] # type: ignore

        if dtype in ["int64", "Int64"]:
            assert is_integer(value)
            return SensitiveInt(value, distance=distance, parents=parents, accountant=accountant)
        elif dtype in ["float64", "Float64"]:
            assert is_floating(value)
            return SensitiveFloat(value, distance=distance, parents=parents, accountant=accountant)
        else:
            raise Exception

    def _get_distributed_df(self) -> _pd.DataFrame:
        assert self._distance_group_axes is None

        if self._distributed_df is None:
            effective_max_distance = float(self.distance.max())
            if effective_max_distance != 1.0:
                raise DPError("Parallel composition requires adjacent databases (max_distance=1)")

            distances = self._ensure_partitioned_distances()
            n_children = len(self.index) * len(self.columns)
            child_accountants = iter(self.accountant.create_parallel_accountants(n_children))

            ddf = _pd.DataFrame(index=self.index, columns=self.columns)

            for col, dc in zip(self.columns, distances):
                for idx, d in zip(self.index, dc.create_exclusive_children(len(self.index))):
                    child_accountant = next(child_accountants)
                    ddf.loc[idx, col] = self._wrap_sensitive_value(self._value.loc[idx, col], column=col, distance=d, parents=[self], # type: ignore
                                                                   accountant=child_accountant)

            self._distributed_df = ddf

        return self._distributed_df

    def _get_dummy_df(self) -> _pd.DataFrame:
        dummy_data = [['***' for _ in self.columns] for _ in self.index]
        return _pd.DataFrame(dummy_data, index=self.index, columns=self.columns)

    def __str__(self) -> str:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_df().__str__()

    def __repr__(self) -> str:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_df().__repr__()

    def _repr_html_(self) -> Any:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_df()._repr_html_() # type: ignore

    def __len__(self) -> int:
        return self.shape[0]

    @egrpc.multimethod
    def __getitem__(self, key: ElementType) -> SensitiveSeries[ElementType]:
        if self._distance_group_axes is None:
            ddf = self._get_distributed_df()
            return SensitiveSeries[ElementType](self._value[key],
                                                distance_group_axes = None,
                                                distance            = self._distance_of(key),
                                                index               = self.index,
                                                name                = key,
                                                parents             = [self],
                                                distributed_ser     = ddf[key])

        elif self._distance_group_axes == (1,):
            return SensitiveSeries[ElementType](self._value[key],
                                                distance_group_axes = None,
                                                distance            = self._distance_of(key),
                                                index               = self.index,
                                                name                = key,
                                                parents             = [self])

        else:
            raise Exception

    @egrpc.property
    def index(self) -> _pd.Index: # type: ignore
        return self._value.index

    @egrpc.property
    def columns(self) -> _pd.Index: # type: ignore
        return self._value.columns

    # FIXME
    @property
    def dtypes(self) -> _pd.Series[Any]:
        return self._value.dtypes

    @egrpc.property
    def name(self) -> str | None:
        return str(self._value.name) if self._value.name is not None else None

    @egrpc.property
    def shape(self) -> tuple[int, int]:
        return self._value.shape

    @egrpc.property
    def size(self) -> int:
        return self._value.size

    @egrpc.method
    def to_numpy(self, copy: bool | None = None) -> SensitiveNDArray:
        if copy is None or not copy:
            raise NotImplementedError("`copy` must be True in privjail to_numpy().")

        try:
            array = self._value.to_numpy(dtype=float, copy=copy)
        except (TypeError, ValueError) as exc:
            raise DPError("All columns must have RealDomain to be converted to NDArray.") from exc

        return SensitiveNDArray(value     = array,
                                distance  = self.distance,
                                norm_type = "l1",
                                parents   = [self])

    @property
    def values(self) -> SensitiveNDArray:
        return self.to_numpy(copy=True)

    def reveal(self,
               *,
               eps   : floating | None = None,
               delta : floating | None = None,
               rho   : floating | None = None,
               scale : floating | None = None,
               mech  : str             = "laplace",
               ) -> _pd.DataFrame:
        if mech == "laplace":
            from ..mechanism import laplace_mechanism
            return laplace_mechanism(self,
                                     eps   = float(eps)   if eps   is not None else None,
                                     scale = float(scale) if scale is not None else None)
        elif mech == "gaussian":
            from ..mechanism import gaussian_mechanism
            return gaussian_mechanism(self,
                                      eps   = float(eps)   if eps   is not None else None,
                                      delta = float(delta) if delta is not None else None,
                                      rho   = float(rho)   if rho   is not None else None,
                                      scale = float(scale) if scale is not None else None)
        else:
            raise ValueError(f"Unknown DP mechanism: '{mech}'")
