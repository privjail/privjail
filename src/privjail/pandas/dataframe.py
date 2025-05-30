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
from ..prisoner import Prisoner, SensitiveInt
from ..distance import Distance
from .util import ElementType, PTag, new_ptag, total_max_distance
from .domain import Domain, BoolDomain, RealDomain, CategoryDomain
from .series import PrivSeries, SensitiveSeries

if TYPE_CHECKING:
    from .groupby import PrivDataFrameGroupBy

T = TypeVar("T")

@egrpc.remoteclass
class PrivDataFrame(Prisoner[_pd.DataFrame]):
    """Private DataFrame.

    Each row in this dataframe object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of rows is treated as a sensitive value.
    """
    _ptag    : PTag
    _domains : Mapping[str, Domain]

    def __init__(self,
                 data         : Any,
                 domains      : Mapping[str, Domain],
                 distance     : Distance,
                 index        : Any                     = None,
                 columns      : Any                     = None,
                 dtype        : Any                     = None,
                 copy         : bool                    = False,
                 *,
                 parents      : Sequence[Prisoner[Any]] = [],
                 root_name    : str | None              = None,
                 preserve_row : bool | None             = None,
                 ):
        if len(parents) == 0:
            preserve_row = False
        elif preserve_row is None:
            raise ValueError("preserve_row is required when parents are specified.")

        self._domains = domains

        if preserve_row:
            parent_tags = [p._ptag for p in parents if isinstance(p, (PrivDataFrame, PrivSeries))]
            assert len(parent_tags) > 0 and all([parent_tags[0] == pt for pt in parent_tags])
            self._ptag = parent_tags[0]
        else:
            self._ptag = new_ptag()

        data = _pd.DataFrame(data, index, columns, dtype, copy)
        if preserve_row:
            super().__init__(value=data, distance=distance, parents=parents, root_name=root_name)
        else:
            super().__init__(value=data, distance=distance, parents=parents, root_name=root_name)

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
        # TODO: how to pass `value_type` from server to client via egrpc?
        return PrivSeries[value_type](data         = self._value.__getitem__(key), # type: ignore[valid-type]
                                      domain       = self.domains[key],
                                      distance     = self.distance,
                                      parents      = [self],
                                      preserve_row = True)

    @__getitem__.register
    def _(self, key: list[str]) -> PrivDataFrame:
        new_domains = {c: d for c, d in self.domains.items() if c in key}
        return PrivDataFrame(data         = self._value.__getitem__(key),
                             domains      = new_domains,
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @__getitem__.register
    def _(self, key: PrivSeries[bool]) -> PrivDataFrame:
        if self._ptag != key._ptag:
            raise DPError("df[bool_vec] cannot be accepted when df and bool_vec are not row-preserved.")

        return PrivDataFrame(data         = self._value.__getitem__(key._value),
                             domains      = self.domains,
                             distance     = self.distance,
                             parents      = [self, key],
                             preserve_row = False)

    @egrpc.multimethod
    def __setitem__(self, key: str, value: ElementType) -> None:
        # TODO: consider domain transform
        self._value[key] = value

    @__setitem__.register
    def _(self, key: str, value: PrivSeries[Any]) -> None:
        new_domains = dict()
        for col, domain in self.domains.items():
            if col == key:
                new_domains[col] = value.domain
            else:
                new_domains[col] = domain
        self._domains = new_domains

        self._value[key] = value._value

    @__setitem__.register
    def _(self, key: list[str], value: ElementType) -> None:
        # TODO: consider domain transform
        self._value[key] = value

    @__setitem__.register
    def _(self, key: list[str], value: PrivDataFrame) -> None:
        # TODO: consider domain transform
        self._value[key] = value._value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: ElementType) -> None:
        # TODO: consider domain transform
        if self._ptag != key._ptag:
            raise DPError("df[bool_vec] cannot be accepted when df and bool_vec are not row-preserved.")

        self._value[key._value] = value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: list[ElementType]) -> None:
        # TODO: consider domain transform
        if self._ptag != key._ptag:
            raise DPError("df[bool_vec] cannot be accepted when df and bool_vec are not row-preserved.")

        self._value[key._value] = value

    @__setitem__.register
    def _(self, key: PrivSeries[bool], value: PrivDataFrame) -> None:
        # TODO: consider domain transform
        if self._ptag != key._ptag:
            raise DPError("df[bool_vec] cannot be accepted when df and bool_vec are not row-preserved.")

        self._value[key._value] = value._value

    @egrpc.multimethod
    def __eq__(self, other: PrivDataFrame) -> PrivDataFrame:
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive dataframes for comparison can be different.")

        return PrivDataFrame(data         = self._value == other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__eq__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        return PrivDataFrame(data         = self._value == other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __ne__(self, other: PrivDataFrame) -> PrivDataFrame:
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive dataframes for comparison can be different.")

        return PrivDataFrame(data         = self._value != other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__ne__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        return PrivDataFrame(data         = self._value != other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __lt__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive dataframes for comparison can be different.")

        return PrivDataFrame(data         = self._value < other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__lt__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        return PrivDataFrame(data         = self._value < other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __le__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive dataframes for comparison can be different.")

        return PrivDataFrame(data         = self._value <= other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__le__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        return PrivDataFrame(data         = self._value <= other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __gt__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive dataframes for comparison can be different.")

        return PrivDataFrame(data         = self._value > other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__gt__.register
    def _(self, other: ElementType) -> PrivDataFrame:
        return PrivDataFrame(data         = self._value > other,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = True)

    @egrpc.multimethod
    def __ge__(self, other: PrivDataFrame) -> PrivDataFrame: # type: ignore
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive dataframes for comparison can be different.")

        return PrivDataFrame(data         = self._value >= other._value,
                             domains      = {c: BoolDomain() for c in self.domains},
                             distance     = self.distance,
                             parents      = [self, other],
                             preserve_row = True)

    @__ge__.register
    def _(self, other: ElementType) -> PrivDataFrame:
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
        nrows = SensitiveInt(value=self._value.shape[0], distance=self.distance, parents=[self])
        ncols = self._value.shape[1]
        return (nrows, ncols)

    @egrpc.property
    def size(self) -> SensitiveInt:
        return SensitiveInt(value=self._value.size, distance=self.distance * len(self._value.columns), parents=[self])

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

    # TODO: add test
    @egrpc.method
    def head(self, n: int = 5) -> PrivDataFrame:
        return PrivDataFrame(data         = self._value.head(n),
                             domains      = self.domains,
                             distance     = self.distance * 2,
                             parents      = [self],
                             preserve_row = False)

    # TODO: add test
    @egrpc.method
    def tail(self, n: int = 5) -> PrivDataFrame:
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
             ) -> PrivDataFrame | None:
        if axis not in (1, "columns") or index is not None:
            raise DPError("Rows cannot be dropped")

        if isinstance(labels, str):
            new_domains = {k: v for k, v in self.domains.items() if k != labels}
        elif isinstance(labels, list):
            new_domains = {k: v for k, v in self.domains.items() if k not in labels}
        else:
            raise TypeError

        if inplace:
            self._value.drop(labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace) # type: ignore
            self._domains = new_domains
            return None
        else:
            return PrivDataFrame(data         = self._value.drop(labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace), # type: ignore
                                 domains      = new_domains,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = True)

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
            self._ptag = new_ptag()
            return None
        else:
            return PrivDataFrame(data         = self._value.sort_values(by, ascending=ascending, inplace=inplace, kind="stable"),
                                 domains      = self.domains,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = False)

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
            return PrivDataFrame(data         = self._value.dropna(inplace=inplace),
                                 domains      = new_domains,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = True)

    def groupby(self,
                by         : str, # TODO: support more
                level      : int | None                   = None, # TODO: support multiindex?
                as_index   : bool                         = True,
                sort       : bool                         = True,
                group_keys : bool                         = True,
                observed   : bool                         = True,
                dropna     : bool                         = True,
                keys       : Sequence[ElementType] | None = None, # extra argument for privjail
                ) -> PrivDataFrameGroupBy:
        result = self._groupby_impl(by, level=level, as_index=as_index, sort=sort,
                                    group_keys=group_keys, observed=observed, dropna=dropna, keys=keys)

        from .groupby import PrivDataFrameGroupBy
        return PrivDataFrameGroupBy(result, [by])

    @egrpc.method
    def _groupby_impl(self,
                      by         : str, # TODO: support more
                      level      : int | None                   = None, # TODO: support multiindex?
                      as_index   : bool                         = True,
                      sort       : bool                         = True,
                      group_keys : bool                         = True,
                      observed   : bool                         = True,
                      dropna     : bool                         = True,
                      keys       : Sequence[ElementType] | None = None, # extra argument for privjail
                      ) -> dict[ElementType, PrivDataFrame]:
        key_domain = self.domains[by]
        if isinstance(key_domain, CategoryDomain):
            keys = key_domain.categories

        if keys is None:
            raise DPError("Please provide the `keys` argument to prevent privacy leakage for non-categorical columns.")

        # TODO: consider extra arguments
        grouped = self._value.groupby(by, observed=observed)

        # set empty groups for absent `keys`
        columns = self._value.columns
        dtypes = self._value.dtypes
        def gen_empty_df() -> _pd.DataFrame:
            return _pd.DataFrame({c: _pd.Series(dtype=d) for c, d in zip(columns, dtypes)})
        groups = {key: grouped.get_group(key) if key in grouped.groups else gen_empty_df() for key in keys}

        # create new child distance variables to express exclusiveness
        distances = self.distance.create_exclusive_distances(len(groups))

        # create a dummy prisoner to track exclusive provenance
        prisoner_dummy = Prisoner(0, self.distance, parents=[self], children_type="exclusive")

        # wrap each group by PrivDataFrame
        # TODO: update childrens' category domain that is chosen for the groupby key
        return {key: PrivDataFrame(df, domains=self.domains, distance=d, parents=[prisoner_dummy], preserve_row=False) \
                for (key, df), d in zip(groups.items(), distances)}

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
