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
from typing import overload, TypeVar, Any, Literal, Sequence, Mapping, TYPE_CHECKING
import copy

import numpy as _np
import pandas as _pd

from .. import egrpc
from ..util import DPError, is_realnum, realnum, floating
from ..prisoner import SensitiveInt
from ..realexpr import RealExpr
from .util import ElementType, PrivPandasBase, assert_ptag, total_max_distance
from .domain import Domain, BoolDomain, RealDomain, CategoryDomain
from .series import PrivSeries, SensitiveSeries

if TYPE_CHECKING:
    from .groupby import PrivDataFrameGroupBy, PrivDataFrameGroupByUser

T = TypeVar("T")

@egrpc.remoteclass
class PrivDataFrame(PrivPandasBase[_pd.DataFrame]):
    """Private DataFrame.

    Each row in this dataframe object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of rows is treated as a sensitive value.
    """
    _domains       : Mapping[str, Domain]
    _user_key      : str | None
    _user_max_freq : RealExpr

    def __init__(self,
                 data          : Any,
                 domains       : Mapping[str, Domain],
                 distance      : RealExpr,
                 user_key      : str | None                    = None,
                 user_max_freq : RealExpr                      = RealExpr.INF,
                 index         : Any                           = None,
                 columns       : Any                           = None,
                 dtype         : Any                           = None,
                 copy          : bool                          = False,
                 *,
                 parents       : Sequence[PrivPandasBase[Any]] = [],
                 root_name     : str | None                    = None,
                 preserve_row  : bool | None                   = None,
                 ):
        self._domains = domains
        self._user_key = user_key
        self._user_max_freq = user_max_freq
        df = _pd.DataFrame(data, index, columns, dtype, copy)
        assert user_key is None or user_key in df.columns
        super().__init__(value=df, distance=distance, parents=parents, root_name=root_name, preserve_row=preserve_row)

    def _is_eldp(self) -> bool:
        return self._user_key is None

    def _is_uldp(self) -> bool:
        return self._user_key is not None

    def _assert_not_uldp(self) -> None:
        if self._is_uldp():
            raise DPError("This operation is not permitted for user DataFrame.")

    def _assert_user_key_not_in(self, columns: list[str]) -> None:
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
    def __getitem__(self, key: str) -> PrivSeries[ElementType]:
        # TODO: consider duplicated column names
        value_type = self.domains[key].type()
        user_key_included = self._user_key == key
        # TODO: how to pass `value_type` from server to client via egrpc?
        return PrivSeries[value_type](data          = self._value.__getitem__(key), # type: ignore[valid-type]
                                      domain        = self.domains[key],
                                      distance      = self.distance if user_key_included else self._eldp_distance(),
                                      is_user_key   = user_key_included,
                                      user_max_freq = self._user_max_freq if user_key_included else RealExpr.INF,
                                      parents       = [self],
                                      preserve_row  = True)

    @__getitem__.register
    def _(self, key: list[str]) -> PrivDataFrame:
        new_domains = {c: d for c, d in self.domains.items() if c in key}
        user_key_included = self._user_key in key
        return PrivDataFrame(data          = self._value.__getitem__(key),
                             domains       = new_domains,
                             distance      = self.distance if user_key_included else self._eldp_distance(),
                             user_key      = self._user_key if user_key_included else None,
                             user_max_freq = self._user_max_freq if user_key_included else RealExpr.INF,
                             parents       = [self],
                             preserve_row  = True)

    @__getitem__.register
    def _(self, key: PrivSeries[bool]) -> PrivDataFrame:
        assert_ptag(self, key)
        return PrivDataFrame(data          = self._value.__getitem__(key._value),
                             domains       = self.domains,
                             distance      = self.distance,
                             user_key      = self._user_key,
                             user_max_freq = self._user_max_freq,
                             parents       = [self, key],
                             preserve_row  = False)

    @egrpc.multimethod
    def __setitem__(self, key: str, value: ElementType) -> None:
        self._assert_user_key_not_in([key])
        # TODO: consider domain transform
        self._value[key] = value

    @__setitem__.register
    def _(self, key: str, value: PrivSeries[Any]) -> None:
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
    def _(self, key: list[str], value: ElementType) -> None:
        # TODO: consider domain transform
        self._assert_user_key_not_in(key)
        self._value[key] = value

    @__setitem__.register
    def _(self, key: list[str], value: PrivDataFrame) -> None:
        # TODO: consider domain transform
        self._assert_user_key_not_in(key)
        value._assert_not_uldp()
        self._value[key] = value._value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: ElementType) -> None:
        # TODO: consider domain transform
        assert_ptag(self, key)
        self._assert_not_uldp()
        self._value[key._value] = value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: list[ElementType]) -> None:
        # TODO: consider domain transform
        assert_ptag(self, key)
        self._assert_not_uldp()
        self._value[key._value] = value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: PrivDataFrame) -> None:
        # TODO: consider domain transform
        assert_ptag(self, key)
        self._assert_not_uldp()
        value._assert_not_uldp()
        self._value[key._value] = value._value

    @egrpc.multimethod
    def __eq__(self, other: PrivDataFrame) -> PrivDataFrame:
        assert_ptag(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data         = self._value == other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__eq__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data         = self._value == other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __ne__(self, other: PrivDataFrame) -> PrivDataFrame:
        assert_ptag(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data         = self._value != other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__ne__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data         = self._value != other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __lt__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        assert_ptag(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data         = self._value < other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__lt__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data         = self._value < other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __le__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        assert_ptag(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data         = self._value <= other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__le__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data         = self._value <= other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __gt__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        assert_ptag(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data         = self._value > other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__gt__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data         = self._value > other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __ge__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        assert_ptag(self, other)
        self._assert_not_uldp()
        other._assert_not_uldp()
        return PrivDataFrame(data         = self._value >= other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__ge__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data         = self._value >= other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.property
    def max_distance(self) -> realnum:
        return self.distance.max()

    @egrpc.property
    def shape(self) -> tuple[SensitiveInt, int]:
        nrows = SensitiveInt(value=self._value.shape[0], distance=self._eldp_distance(), parents=[self])
        ncols = self._value.shape[1]
        return (nrows, ncols)

    @egrpc.property
    def size(self) -> SensitiveInt:
        return SensitiveInt(value=self._value.size, distance=self._eldp_distance() * len(self._value.columns), parents=[self])

    # TODO: define privjail's own Index[T] type
    @property
    def columns(self) -> _pd.Index[str]:
        return _pd.Index(self._get_columns())

    @egrpc.method
    def _get_columns(self) -> list[str]:
        return list(self._value.columns)

    # FIXME
    @property
    def dtypes(self) -> _pd.Series[Any]:
        return self._value.dtypes

    @egrpc.property
    def domains(self) -> Mapping[str, Domain]:
        return self._domains

    @egrpc.property
    def user_key(self) -> str | None:
        return self._user_key

    @egrpc.property
    def user_max_freq(self) -> int | None:
        return int(self._user_max_freq.max()) if not self._user_max_freq.is_inf() else None

    # TODO: add test
    @egrpc.method
    def head(self, n: int = 5) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data         = self._value.head(n),
                             domains      = self.domains,
                             distance     = self.distance * 2,
                             parents      = [self],
                             preserve_row = False)

    # TODO: add test
    @egrpc.method
    def tail(self, n: int = 5) -> PrivDataFrame:
        self._assert_not_uldp()
        return PrivDataFrame(data         = self._value.tail(n),
                             domains      = self.domains,
                             distance     = self.distance * 2,
                             parents      = [self],
                             preserve_row = False)

    @overload
    def drop(self,
             labels  : str | list[str] | None = ...,
             *,
             axis    : int | str              = ...,
             index   : str | list[str] | None = ...,
             columns : str | list[str] | None = ...,
             level   : int | None             = ...,
             inplace : Literal[True],
             errors  : str                    = "raise",
             ) -> None: ...

    @overload
    def drop(self,
             labels  : str | list[str] | None = ...,
             *,
             axis    : int | str              = ...,
             index   : str | list[str] | None = ...,
             columns : str | list[str] | None = ...,
             level   : int | None             = ...,
             inplace : Literal[False]         = ...,
             errors  : str                    = "raise",
             ) -> PrivDataFrame: ...

    @egrpc.method
    def drop(self,
             labels  : str | list[str] | None = None,
             *,
             axis    : int | str              = 0, # 0, 1, "index", "columns"
             index   : str | list[str] | None = None,
             columns : str | list[str] | None = None,
             level   : int | None             = None,
             inplace : bool                   = False,
             errors  : str                    = "raise", # "raise" | "ignore"
             ) -> PrivDataFrame | None:
        if axis not in (1, "columns") or index is not None:
            raise DPError("Rows cannot be dropped")

        if isinstance(labels, str):
            drop_columns = [labels]
        elif isinstance(labels, list):
            drop_columns = labels
        else:
            raise TypeError

        new_domains = {k: v for k, v in self.domains.items() if k not in drop_columns}
        user_key_included = self._is_uldp() and self._user_key not in drop_columns

        if inplace:
            self._value.drop(labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors) # type: ignore
            self._domains = new_domains
            if not user_key_included:
                self._distance = self._eldp_distance()
                self._user_key = None
                self._user_max_freq = RealExpr.INF
            return None
        else:
            return PrivDataFrame(data          = self._value.drop(labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors), # type: ignore
                                 domains       = new_domains,
                                 distance      = self.distance if user_key_included else self._eldp_distance(),
                                 user_key      = self._user_key if user_key_included else None,
                                 user_max_freq = self._user_max_freq if user_key_included else RealExpr.INF,
                                 parents       = [self],
                                 preserve_row  = True)

    @overload
    def sort_values(self,
                    by        : str | list[str],
                    *,
                    ascending : bool = ...,
                    inplace   : Literal[True],
                    ) -> None: ...

    @overload
    def sort_values(self,
                    by        : str | list[str],
                    *,
                    ascending : bool = ...,
                    inplace   : Literal[False] = ...,
                    ) -> PrivDataFrame: ...

    # TODO: add test
    @egrpc.method
    def sort_values(self,
                    by        : str | list[str],
                    *,
                    ascending : bool = True,
                    inplace   : bool = False,
                    ) -> PrivDataFrame | None:
        if inplace:
            self._value.sort_values(by, ascending=ascending, inplace=inplace, kind="stable")
            self.renew_ptag()
            return None
        else:
            return PrivDataFrame(data          = self._value.sort_values(by, ascending=ascending, inplace=inplace, kind="stable"),
                                 domains       = self.domains,
                                 distance      = self.distance,
                                 user_key      = self._user_key,
                                 user_max_freq = self._user_max_freq,
                                 parents       = [self],
                                 preserve_row  = False)

    @overload
    def replace(self,
                to_replace : ElementType | None = ...,
                value      : ElementType | None = ...,
                *,
                inplace    : Literal[True],
                ) -> None: ...

    @overload
    def replace(self,
                to_replace : ElementType | None = ...,
                value      : ElementType | None = ...,
                *,
                inplace    : Literal[False] = ...,
                ) -> PrivDataFrame: ...

    @egrpc.method
    def replace(self,
                to_replace : ElementType | None = None,
                value      : ElementType | None = None,
                *,
                inplace    : bool = False,
                ) -> PrivDataFrame | None:
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

        if inplace:
            self._value.replace(to_replace, value, inplace=inplace) # type: ignore[arg-type]
            self._domains = new_domains
            return None
        else:
            return PrivDataFrame(data         = self._value.replace(to_replace, value, inplace=inplace), # type: ignore[arg-type]
                                 domains      = new_domains,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = True)

    @overload
    def dropna(self,
               *,
               inplace      : Literal[True],
               ignore_index : bool = ...,
               ) -> None: ...

    @overload
    def dropna(self,
               *,
               inplace      : Literal[False] = ...,
               ignore_index : bool = ...,
               ) -> PrivDataFrame: ...

    @egrpc.method
    def dropna(self,
               *,
               inplace      : bool = False,
               ignore_index : bool = False,
               ) -> PrivDataFrame | None:
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

        if inplace:
            self._value.dropna(inplace=inplace)
            self._domains = new_domains
            return None
        else:
            return PrivDataFrame(data          = self._value.dropna(inplace=inplace),
                                 domains       = new_domains,
                                 distance      = self.distance,
                                 user_key      = self._user_key,
                                 user_max_freq = self._user_max_freq,
                                 parents       = [self],
                                 preserve_row  = True)

    def groupby(self,
                by         : str | list[str],
                level      : int | None      = None, # TODO: support multiindex?
                as_index   : bool            = True,
                sort       : bool            = True,
                group_keys : bool            = True,
                observed   : bool            = True,
                dropna     : bool            = True,
                ) -> PrivDataFrameGroupBy | PrivDataFrameGroupByUser:
        from .groupby import _do_group_by
        return _do_group_by(self, by, level=level, as_index=as_index, sort=sort,
                            group_keys=group_keys, observed=observed, dropna=dropna)

    def sum(self) -> SensitiveSeries[int] | SensitiveSeries[float]:
        data = [self[col].sum() for col in self.columns]
        if all(domain.dtype in ("int64", "Int64") for domain in self.domains.values()):
            return SensitiveSeries[int](data, index=self.columns)
        else:
            return SensitiveSeries[float](data, index=self.columns)

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
        return PrivDataFrame(data          = self._value.sample(frac=frac),
                             domains       = self.domains,
                             distance      = self.distance,
                             user_key      = self._user_key,
                             user_max_freq = self._user_max_freq,
                             parents       = [self],
                             preserve_row  = False)

class SensitiveDataFrame(_pd.DataFrame):
    """Sensitive DataFrame.

    Each value in this dataframe object is considered a sensitive value.
    The numbers of rows and columns are not sensitive.
    This is typically created by counting queries like `pandas.crosstab()` and `pandas.pivot_table()`.
    """
    def max_distance(self) -> realnum:
        return total_max_distance(list(self.values.flatten()))

    def reveal(self, eps: floating, mech: str = "laplace") -> float:
        if mech == "laplace":
            from ..mechanism import laplace_mechanism
            result: float = laplace_mechanism(self, eps)
            return result
        else:
            raise ValueError(f"Unknown DP mechanism: '{mech}'")
