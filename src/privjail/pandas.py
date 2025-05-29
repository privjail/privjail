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
from typing import overload, TypeVar, Any, Literal, Iterator, Generic, Sequence, Mapping, TYPE_CHECKING
from abc import ABC, abstractmethod
import warnings
import json
import copy
import itertools
from dataclasses import dataclass, field

import numpy as _np
import pandas as _pd

from .util import DPError, is_realnum, realnum, floating
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat, _max as smax, _min as smin
from .distance import Distance
from . import egrpc

T = TypeVar("T")

ElementType = realnum | str | bool

PTag = int

ptag_count = 0

def new_ptag() -> PTag:
    global ptag_count
    ptag_count += 1
    return ptag_count

def normalize_column_schema(col_schema: dict[str, Any]) -> dict[str, Any]:
    if "type" not in col_schema:
        raise ValueError("Column schema must have the 'type' field.")

    col_type = col_schema["type"]
    new_col_schema = dict(type=col_type, na_value=col_schema.get("na_value", ""))

    if col_type in ["int64", "Int64", "float64", "Float64"]:
        if "range" not in col_schema:
            new_col_schema["range"] = [None, None]

        else:
            expected_type = int if col_type in ["int64", "Int64"] else float
            if not isinstance(col_schema["range"], list) or len(col_schema["range"]) != 2 or \
                    (col_schema["range"][0] is not None and not isinstance(col_schema["range"][0], expected_type)) or \
                    (col_schema["range"][1] is not None and not isinstance(col_schema["range"][1], expected_type)):
                raise ValueError(f"The 'range' field must be in the form [a, b], where a and b is {expected_type} or null.")

            new_col_schema["range"] = col_schema["range"]

    elif col_type == "category":
        if "categories" not in col_schema:
            raise ValueError("Please specify the 'categories' field for the column of type 'category'.")

        if not isinstance(col_schema["categories"], list) or not all(isinstance(x, str) for x in col_schema["categories"]):
            raise ValueError("The 'categories' field must be a list of strings.")

        if len(col_schema["categories"]) == 0:
            raise ValueError("The 'categories' list must have at least one element.")

        new_col_schema["categories"] = col_schema["categories"]

    elif col_type == "string":
        pass

    else:
        raise ValueError(f"Type '{col_type}' is not supported for dataframe column types.")

    return new_col_schema

def apply_column_schema(ser: _pd.Series[Any], col_schema: dict[str, Any], col_name: str) -> _pd.Series[Any]:
    ser = ser.replace(col_schema["na_value"], None)

    if col_schema["type"] == "int64":
        try:
            return ser.astype("int64")
        except _pd.errors.IntCastingNaNError:
            raise ValueError(f"Column '{col_name}' may include NaN or inf values. Consider specifying 'Int64' for the column type.")

    elif col_schema["type"] == "category":
        category_dtype = _pd.api.types.CategoricalDtype(categories=col_schema["categories"])
        return ser.astype(category_dtype)

    else:
        return ser.astype(col_schema["type"]) # type: ignore[no-any-return]

def column_schema2domain(col_schema: dict[str, Any]) -> Domain:
    col_type = col_schema["type"]

    if col_type in ["int64", "Int64", "float64", "Float64"]:
        return RealDomain(dtype=col_type, range=col_schema["range"])

    elif col_type == "category":
        return CategoryDomain(categories=col_schema["categories"])

    elif col_type == "string":
        return StrDomain()

    else:
        raise RuntimeError

# suppress mypy errors
if TYPE_CHECKING:
    @dataclass
    class Domain(ABC):
        dtype: str

        @abstractmethod
        def type(self) -> type:
            pass

    @dataclass
    class BoolDomain(Domain):
        dtype: str = "bool"

        def type(self) -> type:
            return bool

    @dataclass
    class RealDomain(Domain):
        range: tuple[realnum | None, realnum | None]

        def type(self) -> Any:
            return int if self.dtype in ("int64", "Int64") else float

    @dataclass
    class StrDomain(Domain):
        dtype: str = "string"

        def type(self) -> type:
            return str

    @dataclass
    class CategoryDomain(Domain):
        dtype: str = "categories"
        categories: list[ElementType] = field(default_factory=list)

        def type(self) -> type:
            return str

    @dataclass
    class PrivDataFrameGroupBy:
        # TODO: groups are ordered?
        groups    : Mapping[ElementType, PrivDataFrame]
        key_names : list[str]

        def __len__(self) -> int:
            return len(self.groups)

        def __iter__(self) -> Iterator[tuple[Any, PrivDataFrame]]:
            return iter(self.groups.items())

        def get_group(self, key: Any) -> PrivDataFrame:
            return self.groups[key]

        def sum(self) -> SensitiveDataFrame:
            data = [df.drop(self.key_names, axis=1).sum() for key, df in self.groups.items()]
            return SensitiveDataFrame(data, index=self.groups.keys()) # type: ignore

        def mean(self, eps: float) -> _pd.DataFrame:
            data = [df.drop(self.key_names, axis=1).mean(eps=eps) for key, df in self.groups.items()]
            return _pd.DataFrame(data, index=self.groups.keys()) # type: ignore

else:
    @egrpc.dataclass
    class Domain(ABC):
        dtype: str

        @abstractmethod
        def type(self) -> type:
            pass

    @egrpc.dataclass
    class BoolDomain(Domain):
        dtype: str = "bool"

        def type(self) -> type:
            return bool

    @egrpc.dataclass
    class RealDomain(Domain):
        range: tuple[realnum | None, realnum | None]

        def type(self) -> Any:
            return int if self.dtype in ("int64", "Int64") else float

    @egrpc.dataclass
    class StrDomain(Domain):
        dtype: str = "string"

        def type(self) -> type:
            return str

    @egrpc.dataclass
    class CategoryDomain(Domain):
        dtype: str = "categories"
        categories: list[ElementType] = field(default_factory=list)

        def type(self) -> type:
            assert len(self.categories) > 0
            return type(self.categories[0]) # TODO: how about other elements?

    @egrpc.dataclass
    class PrivDataFrameGroupBy:
        # TODO: groups are ordered?
        groups    : Mapping[ElementType, PrivDataFrame]
        key_names : list[str]

        def __len__(self) -> int:
            return len(self.groups)

        def __iter__(self) -> Iterator[tuple[Any, PrivDataFrame]]:
            return iter(self.groups.items())

        def get_group(self, key: Any) -> PrivDataFrame:
            return self.groups[key]

        def sum(self) -> SensitiveDataFrame:
            data = [df.drop(self.key_names, axis=1).sum() for key, df in self.groups.items()]
            return SensitiveDataFrame(data, index=self.groups.keys())

        def mean(self, eps: float) -> _pd.DataFrame:
            data = [df.drop(self.key_names, axis=1).mean(eps=eps) for key, df in self.groups.items()]
            return _pd.DataFrame(data, index=self.groups.keys())

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

    @egrpc.method
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
        key_domain = self.domains[by]
        if isinstance(key_domain, CategoryDomain):
            keys = key_domain.categories

        if keys is None:
            raise DPError("Please provide the `keys` argument to prevent privacy leakage for non-categorical columns.")

        # TODO: consider extra arguments
        groupby_obj = self._value.groupby(by, observed=observed)

        # exclude groups not included in the specified `keys`
        groups = {key: df for key, df in groupby_obj if key in keys}

        # include empty groups for absent `keys` and sort by `keys`
        columns = self._value.columns
        dtypes = self._value.dtypes
        groups = {key: groups.get(key, _pd.DataFrame({c: _pd.Series(dtype=d) for c, d in zip(columns, dtypes)})) for key in keys} # type: ignore

        # create new child distance variables to express exclusiveness
        distances = self.distance.create_exclusive_distances(len(groups))

        # create a dummy prisoner to track exclusive provenance
        prisoner_dummy = Prisoner(0, self.distance, parents=[self], children_type="exclusive")

        # wrap each group by PrivDataFrame
        # TODO: update childrens' category domain that is chosen for the groupby key
        priv_groups = {key: PrivDataFrame(df, domains=self.domains, distance=d, parents=[prisoner_dummy], preserve_row=False) for (key, df), d in zip(groups.items(), distances)}

        return PrivDataFrameGroupBy(priv_groups, [by]) # type: ignore[arg-type]

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

# to avoid TypeError: type 'Series' is not subscriptable
# class PrivSeries(Prisoner[_pd.Series[T]]):
@egrpc.remoteclass
class PrivSeries(Generic[T], Prisoner[_pd.Series]): # type: ignore[type-arg]
    """Private Series.

    Each value in this series object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of values is treated as a sensitive value.
    """
    _domain : Domain
    _ptag   : PTag

    def __init__(self,
                 data         : Any,
                 domain       : Domain,
                 distance     : Distance,
                 *,
                 parents      : Sequence[Prisoner[Any]] = [],
                 root_name    : str | None              = None,
                 preserve_row : bool | None             = None,
                 **kwargs    : Any,
                 ):
        if len(parents) == 0:
            preserve_row = False
        elif preserve_row is None:
            raise ValueError("preserve_row is required when parents are specified.")

        self._domain = domain

        if preserve_row:
            parent_tags = [p._ptag for p in parents if isinstance(p, (PrivDataFrame, PrivSeries))]
            assert len(parent_tags) > 0 and all([parent_tags[0] == pt for pt in parent_tags])
            self._ptag = parent_tags[0]
        else:
            self._ptag = new_ptag()

        data = _pd.Series(data, **kwargs)
        super().__init__(value=data, distance=distance, parents=parents, root_name=root_name)

    def _get_dummy_ser(self, n_rows: int = 3) -> _pd.Series[str]:
        index = list(range(n_rows)) + ['...']
        dummy_data = ['***' for _ in range(n_rows)] + ['...']
        # TODO: dtype becomes 'object'
        return _pd.Series(dummy_data, index=index, name=self.name)

    def __str__(self) -> str:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_ser().__str__().replace("dtype: object", f"dtype: {self.domain.dtype}")

    def __repr__(self) -> str:
        with _pd.option_context('display.show_dimensions', False):
            return self._get_dummy_ser().__repr__().replace("dtype: object", f"dtype: {self.domain.dtype}")

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(ser) is not supported. Use ser.shape[0] or ser.size instead.")

    @egrpc.multimethod
    def __getitem__(self, key: PrivSeries[bool]) -> PrivSeries[T]:
        if self._ptag != key._ptag:
            raise DPError("ser[bool_vec] cannot be accepted when ser and bool_vec are not row-preserved.")

        return PrivSeries[T](data         = self._value.__getitem__(key._value),
                             domain       = self.domain,
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = False)

    @egrpc.multimethod
    def __setitem__(self, key: PrivSeries[bool], value: ElementType) -> None:
        if self._ptag != key._ptag:
            raise DPError("df[bool_vec] cannot be accepted when ser and bool_vec are not row-preserved.")

        self._value[key._value] = value

    @egrpc.multimethod
    def __eq__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]:
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive series for comparison can be different.")

        return PrivSeries[bool](data         = self._value == other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__eq__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value == other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __ne__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]:
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive series for comparison can be different.")

        return PrivSeries[bool](data         = self._value != other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__ne__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value != other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __lt__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]: # type: ignore
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive series for comparison can be different.")

        return PrivSeries[bool](data         = self._value < other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__lt__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value < other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __le__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]: # type: ignore
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive series for comparison can be different.")

        return PrivSeries[bool](data         = self._value <= other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__le__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value <= other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __gt__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]: # type: ignore
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive series for comparison can be different.")

        return PrivSeries[bool](data         = self._value > other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__gt__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value > other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.multimethod
    def __ge__(self, other: PrivSeries[ElementType]) -> PrivSeries[bool]: # type: ignore
        if self._ptag != other._ptag:
            raise DPError("Length of sensitive series for comparison can be different.")

        return PrivSeries[bool](data         = self._value >= other._value,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self, other],
                                preserve_row = True)

    @__ge__.register
    def _(self, other: ElementType) -> PrivSeries[bool]:
        return PrivSeries[bool](data         = self._value >= other,
                                domain       = BoolDomain(),
                                distance     = self.distance,
                                parents      = [self],
                                preserve_row = True)

    @egrpc.property
    def max_distance(self) -> realnum:
        return self.distance.max()

    @egrpc.property
    def shape(self) -> tuple[SensitiveInt]:
        nrows = SensitiveInt(value=self._value.shape[0], distance=self.distance, parents=[self])
        return (nrows,)

    @egrpc.property
    def size(self) -> SensitiveInt:
        return SensitiveInt(value=self._value.size, distance=self.distance, parents=[self])

    # FIXME
    @property
    def dtype(self) -> Any:
        return self._value.dtype

    @egrpc.property
    def name(self) -> str | None:
        return str(self._value.name) if self._value.name is not None else None

    @egrpc.property
    def domain(self) -> Domain:
        return self._domain

    # TODO: add test
    @egrpc.method
    def head(self, n: int = 5) -> PrivSeries[T]:
        return PrivSeries[T](data         = self._value.head(n),
                             domain       = self.domain,
                             distance     = self.distance * 2,
                             parents      = [self],
                             preserve_row = False)

    # TODO: add test
    @egrpc.method
    def tail(self, n: int = 5) -> PrivSeries[T]:
        return PrivSeries[T](data         = self._value.tail(n),
                             domain       = self.domain,
                             distance     = self.distance * 2,
                             parents      = [self],
                             preserve_row = False)

    @overload
    def sort_values(self,
                    *,
                    ascending : bool = ...,
                    inplace   : Literal[True],
                    ) -> None: ...

    @overload
    def sort_values(self,
                    *,
                    ascending : bool = ...,
                    inplace   : Literal[False] = ...,
                    ) -> PrivSeries[T]: ...

    # TODO: add test
    @egrpc.method
    def sort_values(self,
                    *,
                    ascending : bool = True,
                    inplace   : bool = False,
                    ) -> PrivSeries[T] | None:
        if inplace:
            self._value.sort_values(ascending=ascending, inplace=inplace, kind="stable")
            self._ptag = new_ptag()
            return None
        else:
            return PrivSeries[T](data         = self._value.sort_values(ascending=ascending, inplace=inplace, kind="stable"),
                                 domain       = self.domain,
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
                ) -> PrivSeries[T]: ...

    @egrpc.method
    def replace(self,
                to_replace : ElementType | None = None,
                value      : ElementType | None = None,
                *,
                inplace    : bool = False,
                ) -> PrivSeries[T] | None:
        if (not is_realnum(to_replace)) or (not is_realnum(value)):
            # TODO: consider string and category dtype
            raise NotImplementedError

        if self.domain.dtype == "int64" and _np.isnan(value):
            new_domain = copy.copy(self.domain)
            new_domain.dtype = "Int64"

        elif isinstance(self.domain, RealDomain):
            a, b = self.domain.range
            if (a is None or a <= to_replace) and (b is None or to_replace <= b):
                new_a = min(a, value) if a is not None else None # type: ignore[type-var]
                new_b = max(b, value) if b is not None else None # type: ignore[type-var]

                new_domain = copy.copy(self.domain)
                new_domain.range = (new_a, new_b)

            else:
                new_domain = self.domain

        else:
            new_domain = self.domain

        if inplace:
            self._value.replace(to_replace, value, inplace=inplace) # type: ignore[arg-type]
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data         = self._value.replace(to_replace, value, inplace=inplace), # type: ignore[arg-type]
                                 domain       = new_domain,
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
               ) -> PrivSeries[T]: ...

    @egrpc.method
    def dropna(self,
               *,
               inplace      : bool = False,
               ignore_index : bool = False,
               ) -> PrivSeries[T] | None:
        if ignore_index:
            raise DPError("`ignore_index` must be False. Index cannot be reindexed with positions.")

        if self.domain.dtype == "Int64":
            new_domain = copy.copy(self.domain)
            new_domain.dtype = "int64"
        else:
            new_domain = self.domain

        if inplace:
            self._value.dropna(inplace=inplace)
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data         = self._value.dropna(inplace=inplace),
                                 domain       = new_domain,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = True)

    @overload
    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : Literal[True],
             ) -> None: ...

    @overload
    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : Literal[False] = ...,
             ) -> PrivSeries[T]: ...

    @egrpc.method
    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : bool = False,
             ) -> PrivSeries[T] | None:
        if not isinstance(self.domain, RealDomain):
            raise TypeError("Domain must be real numbers.")

        new_domain = copy.copy(self.domain)
        a, b = self.domain.range
        new_a = a if lower is None else lower if a is None else max(a, lower) # type: ignore[type-var]
        new_b = b if upper is None else upper if b is None else min(b, upper) # type: ignore[type-var]
        new_domain.range = (new_a, new_b)

        if inplace:
            self._value.clip(lower, upper, inplace=inplace) # type: ignore[arg-type]
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data         = self._value.clip(lower, upper, inplace=inplace), # type: ignore[arg-type]
                                 domain       = new_domain,
                                 distance     = self.distance,
                                 parents      = [self],
                                 preserve_row = True)

    @egrpc.method
    def sum(self) -> SensitiveInt | SensitiveFloat:
        if not isinstance(self.domain, RealDomain):
            raise TypeError("Domain must be real numbers.")

        if None in self.domain.range:
            raise DPError("The range is unbounded. Use clip().")

        a, b = self.domain.range

        if a is None or b is None:
            raise DPError("The range is unbounded. Use clip().")

        new_distance = self.distance * max(_np.abs(a), _np.abs(b))

        s = self._value.sum()

        if self.domain.dtype in ["int64", "Int64"]:
            return SensitiveInt(s, new_distance, parents=[self])
        elif self.domain.dtype in ["float64", "Float64"]:
            return SensitiveFloat(s, new_distance, parents=[self])
        else:
            raise ValueError

    @egrpc.method
    def mean(self, eps: float) -> float:
        if not isinstance(self.domain, RealDomain):
            raise TypeError("Domain must be real numbers.")

        if eps <= 0:
            raise DPError(f"Invalid epsilon ({eps})")

        a, b = self.domain.range

        if a is None or b is None:
            raise DPError("The range is unbounded. Use clip().")

        sum_sensitivity = (self.distance * max(_np.abs(a), _np.abs(b))).max()
        count_sensitivity = self.distance.max()

        self.consume_privacy_budget(eps)

        s = _np.random.laplace(loc=float(self._value.sum()), scale=float(sum_sensitivity) / (eps / 2))
        c = _np.random.laplace(loc=float(self._value.shape[0]), scale=float(count_sensitivity) / (eps / 2))

        return s / c

    def value_counts(self,
                     normalize : bool                     = False,
                     sort      : bool                     = True,
                     ascending : bool                     = False,
                     bins      : int | None               = None,
                     dropna    : bool                     = True,
                     values    : list[ElementType] | None = None, # extra argument for privjail
                     ) -> SensitiveSeries[int]:
        # TODO: make SensitiveSeries a dataclass
        result = self._value_counts_impl(normalize, sort, ascending, bins, dropna, values)
        return SensitiveSeries[int](data=list(result.values()), index=list(result.keys()), dtype="object")

    @egrpc.method
    def _value_counts_impl(self,
                           normalize : bool                     = False,
                           sort      : bool                     = True,
                           ascending : bool                     = False,
                           bins      : int | None               = None,
                           dropna    : bool                     = True,
                           values    : list[ElementType] | None = None, # extra argument for privjail
                           ) -> dict[ElementType, SensitiveInt]:
        if normalize:
            # TODO: what is the sensitivity?
            raise NotImplementedError

        if bins is not None:
            # TODO: support continuous values
            raise NotImplementedError

        if sort:
            raise DPError("The `sort` argument must be False.")

        if isinstance(self.domain, CategoryDomain):
            values = self.domain.categories

        if values is None:
            raise DPError("Please provide the `values` argument to prevent privacy leakage.")

        if isinstance(values, Prisoner):
            raise DPError("`values` cannot be sensitive values.")

        if not dropna and not any(_np.isnan(values)): # type: ignore
            # TODO: consider handling for pd.NA
            warnings.warn("Counts for NaN will be dropped from the result because NaN is not included in `values`", UserWarning)

        counts = self._value.value_counts(normalize, sort, ascending, bins, dropna)

        # Select only the specified values and fill non-existent counts with 0
        counts = counts.reindex(values).fillna(0).astype(int)

        distances = self.distance.create_exclusive_distances(counts.size)

        prisoner_dummy = Prisoner(0, self.distance, parents=[self], children_type="exclusive")

        return {k: SensitiveInt(counts.loc[k], distance=distances[i], parents=[prisoner_dummy])
                for i, k in enumerate(counts.index)}

@egrpc.function
def total_max_distance(prisoners: list[SensitiveInt | SensitiveFloat]) -> realnum:
    return sum([x.distance for x in prisoners], start=Distance(0)).max()

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
            from .mechanism import laplace_mechanism
            result: float = laplace_mechanism(self, eps)
            return result
        else:
            raise ValueError(f"Unknown DP mechanism: '{mech}'")

# to avoid TypeError: type 'Series' is not subscriptable
# class SensitiveSeries(_pd.Series[T]):
class SensitiveSeries(Generic[T], _pd.Series): # type: ignore[type-arg]
    """Sensitive Series.

    Each value in this series object is considered a sensitive value.
    The numbers of values are not sensitive.
    This is typically created by counting queries like `PrivSeries.value_counts()`.
    """
    def max_distance(self) -> realnum:
        return total_max_distance(list(self.values))

    def max(self, *args: Any, **kwargs: Any) -> SensitiveInt | SensitiveFloat:
        # TODO: args?
        return smax(self)

    def min(self, *args: Any, **kwargs: Any) -> SensitiveInt | SensitiveFloat:
        # TODO: args?
        return smin(self)

    def reveal(self, eps: floating, mech: str = "laplace") -> float:
        if mech == "laplace":
            from .mechanism import laplace_mechanism
            result: float = laplace_mechanism(self, eps)
            return result
        else:
            raise ValueError(f"Unknown DP mechanism: '{mech}'")

@egrpc.function
def read_csv(filepath: str, schemapath: str | None = None) -> PrivDataFrame:
    # TODO: more vaildation for the input data
    df = _pd.read_csv(filepath)

    if schemapath is not None:
        with open(schemapath, "r") as f:
            schema = json.load(f)
    else:
        schema = dict()

    domains = dict()
    for col in df.columns:
        if not isinstance(col, str):
            raise ValueError("Column name must be a string.")

        if col in schema:
            col_schema = schema[col]
        else:
            col_schema = dict(type="string" if df.dtypes[col] == "object" else df.dtypes[col])

        col_schema = normalize_column_schema(col_schema)

        df[col] = apply_column_schema(df[col], col_schema, col)

        domains[col] = column_schema2domain(col_schema)

    return PrivDataFrame(data=df, domains=domains, distance=Distance(1), root_name=filepath)

def crosstab(index        : PrivSeries[ElementType], # TODO: support Sequence[PrivSeries[ElementType]]
             columns      : PrivSeries[ElementType], # TODO: support Sequence[PrivSeries[ElementType]]
             values       : PrivSeries[ElementType] | None = None,
             rownames     : list[str] | None               = None,
             colnames     : list[str] | None               = None,
             rowvalues    : Sequence[ElementType] | None   = None, # extra argument for privjail
             colvalues    : Sequence[ElementType] | None   = None, # extra argument for privjail
             *,
             aggfunc      : None                           = None,
             margins      : bool                           = False,
             margins_name : str                            = "All",
             dropna       : bool                           = True,
             normalize    : bool | str | int               = False, # TODO: support Literal["all", "index", "columns", 0, 1] in egrpc
             ) -> SensitiveDataFrame:
    result = _crosstab_impl(index, columns, values, rownames, colnames, rowvalues, colvalues,
                            aggfunc=aggfunc, margins=margins, margins_name=margins_name,
                            dropna=dropna, normalize=normalize)

    rowvalues, colvalues, data = result
    priv_counts = SensitiveDataFrame(index=rowvalues, columns=colvalues)

    for i, (idx, col) in enumerate(itertools.product(rowvalues, colvalues)):
        priv_counts.loc[idx, col] = data[i] # type: ignore

    return priv_counts

@egrpc.function
def _crosstab_impl(index        : PrivSeries[ElementType], # TODO: support Sequence[PrivSeries[ElementType]]
                   columns      : PrivSeries[ElementType], # TODO: support Sequence[PrivSeries[ElementType]]
                   values       : PrivSeries[ElementType] | None = None,
                   rownames     : list[str] | None               = None,
                   colnames     : list[str] | None               = None,
                   rowvalues    : Sequence[ElementType] | None   = None, # extra argument for privjail
                   colvalues    : Sequence[ElementType] | None   = None, # extra argument for privjail
                   *,
                   aggfunc      : None                           = None,
                   margins      : bool                           = False,
                   margins_name : str                            = "All",
                   dropna       : bool                           = True,
                   normalize    : bool | str | int               = False, # TODO: support Literal["all", "index", "columns", 0, 1] in egrpc
                   ) -> tuple[list[ElementType], list[ElementType], list[SensitiveInt]]:
    if normalize is not False:
        # TODO: what is the sensitivity?
        print(normalize)
        raise NotImplementedError

    if values is not None or aggfunc is not None:
        # TODO: hard to accept arbitrary user functions
        raise NotImplementedError

    if margins:
        # Sensitivity must be managed separately for value counts and margins
        raise DPError("`margins=True` is not supported. Please manually calculate margins after adding noise.")

    if isinstance(index.domain, CategoryDomain):
        rowvalues = index.domain.categories

    if rowvalues is None:
        raise DPError("Please specify `rowvalues` to prevent privacy leakage.")

    if isinstance(columns.domain, CategoryDomain):
        colvalues = columns.domain.categories

    if colvalues is None:
        raise DPError("Please specify `colvalues` to prevent privacy leakage.")

    # if not dropna and (not any(_np.isnan(rowvalues)) or not any(_np.isnan(colvalues))):
    #     # TODO: consider handling for pd.NA
    #     warnings.warn("Counts for NaN will be dropped from the result because NaN is not included in `rowvalues`/`colvalues`", UserWarning)

    if index._ptag != columns._ptag:
        raise DPError("Series in `index` and `columns` must have the same length.")

    counts = _pd.crosstab(index._value, columns._value,
                          values=None, rownames=rownames, colnames=colnames,
                          aggfunc=None, margins=False, margins_name=margins_name,
                          dropna=dropna, normalize=False)

    # Select only the specified values and fill non-existent counts with 0
    counts = counts.reindex(list(rowvalues), axis="index") \
                   .reindex(list(colvalues), axis="columns") \
                   .fillna(0).astype(int)

    distances = index.distance.create_exclusive_distances(counts.size)

    prisoner_dummy = Prisoner(0, index.distance, parents=[index, columns], children_type="exclusive")

    data = [SensitiveInt(counts.loc[idx, col], distance=distances[i], parents=[prisoner_dummy]) # type: ignore
            for i, (idx, col) in enumerate(itertools.product(rowvalues, colvalues))]

    return list(rowvalues), list(colvalues), data

# TODO: change multifunction -> function by type checking in egrpc.function
@egrpc.multifunction
def cut(x              : PrivSeries[Any],
        bins           : list[int] | list[float],
        right          : bool                            = True,
        labels         : list[ElementType] | bool | None = None,
        retbins        : bool                            = False,
        precision      : int                             = 3,
        include_lowest : bool                            = False
        # TODO: add more parameters
        ) -> PrivSeries[Any]:
    ser = _pd.cut(x._value, bins=bins, right=right, labels=labels, retbins=retbins, precision=precision, include_lowest=include_lowest) # type: ignore

    new_domain = CategoryDomain(categories=list(ser.dtype.categories))

    return PrivSeries[Any](ser, domain=new_domain, distance=x.distance, parents=[x], preserve_row=True)
