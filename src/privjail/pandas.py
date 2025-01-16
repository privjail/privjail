from __future__ import annotations
from typing import overload, TypeVar, Any, Literal, Iterator, Generic, Sequence, Mapping
import warnings
import json
import copy

import numpy as _np
import pandas as _pd
from pandas.api.types import is_list_like
from multimethod import multimethod

from .util import DPError, is_realnum, realnum
from .prisoner import Prisoner, SensitiveInt, SensitiveFloat, _max as smax, _min as smin
from .distance import Distance

T = TypeVar("T")

ElementType = realnum | str | bool

PTag = int

ptag_count = 0

def new_ptag() -> PTag:
    global ptag_count
    ptag_count += 1
    return ptag_count

def is_2d_array(x: Any) -> bool:
    return (
        (isinstance(x, _np.ndarray) and x.ndim >= 2) or
        (isinstance(x, list) and all(isinstance(i, list) for i in x))
    )

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
        return RealDomain(type=col_type, range=col_schema["range"])

    elif col_type == "category":
        return CategoryDomain(categories=col_schema["categories"])

    elif col_type == "string":
        return StrDomain()

    else:
        raise RuntimeError

class Domain:
    type: str

    def __init__(self, type: str):
        self.type = type

class BoolDomain(Domain):
    def __init__(self) -> None:
        super().__init__(type="bool")

class RealDomain(Domain):
    range: tuple[realnum | None, realnum | None]

    def __init__(self, type: str, range: tuple[realnum | None, realnum | None]):
        self.range = range
        super().__init__(type=type)

class StrDomain(Domain):
    def __init__(self) -> None:
        super().__init__(type="string")

class CategoryDomain(Domain):
    categories: list[str]

    def __init__(self, categories: list[str]):
        self.categories = categories
        super().__init__(type="categories")

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

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(df) is not supported. Use df.shape[0] instead.")

    @multimethod
    def __getitem__(self, key: str) -> PrivSeries[Any]:
        # TODO: consider duplicated column names
        return PrivSeries[Any](data         = self._value.__getitem__(key),
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

    @multimethod
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
    def _(self, key: PrivSeries[bool], value: Sequence[ElementType]) -> None:
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

    def __eq__(self, other: Any) -> PrivDataFrame: # type: ignore[override]
        new_domains = {c: BoolDomain() for c in self.domains}

        if isinstance(other, PrivDataFrame):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value == other._value, domains=new_domains, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value == other, domains=new_domains, distance=self.distance, parents=[self], preserve_row=True)

    def __ne__(self, other: Any) -> PrivDataFrame: # type: ignore[override]
        new_domains = {c: BoolDomain() for c in self.domains}

        if isinstance(other, PrivDataFrame):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value != other._value, domains=new_domains, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value != other, domains=new_domains, distance=self.distance, parents=[self], preserve_row=True)

    def __lt__(self, other: Any) -> PrivDataFrame:
        new_domains = {c: BoolDomain() for c in self.domains}

        if isinstance(other, PrivDataFrame):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive dataframes for comparison can be different.")
            return PrivDataFrame(data=self._value < other._value, domains=new_domains, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value < other, domains=new_domains, distance=self.distance, parents=[self], preserve_row=True)

    def __le__(self, other: Any) -> PrivDataFrame:
        new_domains = {c: BoolDomain() for c in self.domains}

        if isinstance(other, PrivDataFrame):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value <= other._value, domains=new_domains, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value <= other, domains=new_domains, distance=self.distance, parents=[self], preserve_row=True)

    def __gt__(self, other: Any) -> PrivDataFrame:
        new_domains = {c: BoolDomain() for c in self.domains}

        if isinstance(other, PrivDataFrame):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value > other._value, domains=new_domains, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value > other, domains=new_domains, distance=self.distance, parents=[self], preserve_row=True)

    def __ge__(self, other: Any) -> PrivDataFrame:
        new_domains = {c: BoolDomain() for c in self.domains}

        if isinstance(other, PrivDataFrame):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value >= other._value, domains=new_domains, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value >= other, domains=new_domains, distance=self.distance, parents=[self], preserve_row=True)

    @property
    def shape(self) -> tuple[SensitiveInt, int]:
        nrows = SensitiveInt(value=self._value.shape[0], distance=self.distance, parents=[self])
        ncols = self._value.shape[1]
        return (nrows, ncols)

    @property
    def size(self) -> SensitiveInt:
        return SensitiveInt(value=self._value.size, distance=self.distance * len(self._value.columns), parents=[self])

    @property
    def columns(self) -> _pd.Index[str]:
        return self._value.columns

    @columns.setter
    def columns(self, value: _pd.Index[str]) -> None:
        self._value.columns = value

    @property
    def dtypes(self) -> _pd.Series[Any]:
        return self._value.dtypes

    @property
    def domains(self) -> Mapping[str, Domain]:
        return self._domains

    @overload
    def replace(self,
                to_replace : Any = ...,
                value      : Any = ...,
                *,
                inplace    : Literal[True],
                **kwargs   : Any,
                ) -> None: ...

    @overload
    def replace(self,
                to_replace : Any  = ...,
                value      : Any  = ...,
                *,
                inplace    : bool = ...,
                **kwargs   : Any,
                ) -> PrivDataFrame: ...

    def replace(self,
                to_replace : Any  = None,
                value      : Any  = None,
                *,
                inplace    : bool = False,
                **kwargs   : Any,
                ) -> PrivDataFrame | None:
        if (not is_realnum(to_replace)) or (not is_realnum(value)):
            # TODO: consider string and category dtype
            raise NotImplementedError

        new_domains = dict()
        for col, domain in self.domains.items():
            if domain.type == "int64" and _np.isnan(value):
                new_domain = copy.copy(domain)
                new_domain.type = "Int64"
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
            self._value.replace(to_replace, value, inplace=inplace, **kwargs) # type: ignore[arg-type]
            self._domains = new_domains
            return None
        else:
            return PrivDataFrame(data=self._value.replace(to_replace, value, inplace=inplace, **kwargs), domains=new_domains, distance=self.distance, parents=[self], preserve_row=True) # type: ignore[arg-type]

    @overload
    def dropna(self,
               *,
               inplace      : Literal[True],
               ignore_index : bool = ...,
               **kwargs     : Any,
               ) -> None: ...

    @overload
    def dropna(self,
               *,
               inplace      : bool = ...,
               ignore_index : bool = ...,
               **kwargs     : Any,
               ) -> PrivDataFrame: ...

    def dropna(self,
               *,
               inplace      : bool = False,
               ignore_index : bool = False,
               **kwargs : Any,
               ) -> PrivDataFrame | None:
        if ignore_index:
            raise DPError("`ignore_index` must be False. Index cannot be reindexed with positions.")

        new_domains = dict()
        for col, domain in self.domains.items():
            if domain.type == "Int64":
                new_domain = copy.copy(domain)
                new_domain.type = "int64"
                new_domains[col] = new_domain
            else:
                new_domains[col] = domain

        if inplace:
            self._value.dropna(inplace=inplace, **kwargs)
            self._domains = new_domains
            return None
        else:
            return PrivDataFrame(data=self._value.dropna(inplace=inplace, **kwargs), domains=new_domains, distance=self.distance, parents=[self], preserve_row=True)

    def groupby(self,
                by         : str, # TODO: support more
                level      : Any                                = None,
                as_index   : bool                               = True,
                sort       : bool                               = True,
                group_keys : bool                               = True,
                observed   : bool                               = True,
                dropna     : bool                               = True,
                keys       : list[Any] | _pd.Series[Any] | None = None, # extra argument for privjail
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
        groups = {key: groups.get(key, _pd.DataFrame({c: _pd.Series(dtype=d) for c, d in zip(columns, dtypes)})) for key in keys}

        # create new child distance variables to express exclusiveness
        distances = self.distance.create_exclusive_distances(len(groups))

        # create a dummy prisoner to track exclusive provenance
        prisoner_dummy = Prisoner(0, self.distance, parents=[self], children_type="exclusive")

        # wrap each group by PrivDataFrame
        # TODO: update childrens' category domain that is chosen for the groupby key
        priv_groups = {key: PrivDataFrame(df, domains=self.domains, distance=d, parents=[prisoner_dummy], preserve_row=False) for (key, df), d in zip(groups.items(), distances)}

        return PrivDataFrameGroupBy(priv_groups)

class PrivDataFrameGroupBy:
    def __init__(self, groups: Mapping[Any, PrivDataFrame]):
        # TODO: groups are ordered?
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[tuple[Any, PrivDataFrame]]:
        return iter(self.groups.items())

    def get_group(self, key: Any) -> PrivDataFrame:
        return self.groups[key]

# to avoid TypeError: type 'Series' is not subscriptable
# class PrivSeries(Prisoner[_pd.Series[T]]):
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
                 index        : Any                     = None,
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

        data = _pd.Series(data, index, **kwargs)
        if preserve_row:
            super().__init__(value=data, distance=distance, parents=parents, root_name=root_name)
        else:
            super().__init__(value=data, distance=distance, parents=parents, root_name=root_name)

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(ser) is not supported. Use ser.shape[0] or ser.size instead.")

    @multimethod
    def __getitem__(self, key: PrivSeries[bool]) -> PrivSeries[T]:
        if self._ptag != key._ptag:
            raise DPError("ser[bool_vec] cannot be accepted when ser and bool_vec are not row-preserved.")

        return PrivSeries[T](data         = self._value.__getitem__(key._value),
                             domain       = self.domain,
                             distance     = self.distance,
                             parents      = [self],
                             preserve_row = False)

    @multimethod
    def __setitem__(self, key: PrivSeries[bool], value: ElementType) -> None:
        if self._ptag != key._ptag:
            raise DPError("df[bool_vec] cannot be accepted when ser and bool_vec are not row-preserved.")

        self._value[key._value] = value

    def __eq__(self, other: Any) -> PrivSeries[bool]: # type: ignore[override]
        if isinstance(other, PrivSeries):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value == other._value, domain=BoolDomain(), distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value == other, domain=BoolDomain(), distance=self.distance, parents=[self], preserve_row=True)

    def __ne__(self, other: Any) -> PrivSeries[bool]: # type: ignore[override]
        if isinstance(other, PrivSeries):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value != other._value, domain=BoolDomain(), distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value != other, domain=BoolDomain(), distance=self.distance, parents=[self], preserve_row=True)

    def __lt__(self, other: Any) -> PrivSeries[bool]:
        if isinstance(other, PrivSeries):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value < other._value, domain=BoolDomain(), distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value < other, domain=BoolDomain(), distance=self.distance, parents=[self], preserve_row=True)

    def __le__(self, other: Any) -> PrivSeries[bool]:
        if isinstance(other, PrivSeries):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value <= other._value, domain=BoolDomain(), distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value <= other, domain=BoolDomain(), distance=self.distance, parents=[self], preserve_row=True)

    def __gt__(self, other: Any) -> PrivSeries[bool]:
        if isinstance(other, PrivSeries):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value > other._value, domain=BoolDomain(), distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value > other, domain=BoolDomain(), distance=self.distance, parents=[self], preserve_row=True)

    def __ge__(self, other: Any) -> PrivSeries[bool]:
        if isinstance(other, PrivSeries):
            if self._ptag != other._ptag:
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value >= other._value, domain=BoolDomain(), distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value >= other, domain=BoolDomain(), distance=self.distance, parents=[self], preserve_row=True)

    @property
    def shape(self) -> tuple[SensitiveInt]:
        nrows = SensitiveInt(value=self._value.shape[0], distance=self.distance, parents=[self])
        return (nrows,)

    @property
    def size(self) -> SensitiveInt:
        return SensitiveInt(value=self._value.size, distance=self.distance, parents=[self])

    @property
    def dtypes(self) -> Any:
        return self._value.dtypes

    @property
    def domain(self) -> Domain:
        return self._domain

    @overload
    def replace(self,
                to_replace : Any = ...,
                value      : Any = ...,
                *,
                inplace    : Literal[True],
                **kwargs   : Any,
                ) -> None: ...

    @overload
    def replace(self,
                to_replace : Any  = ...,
                value      : Any  = ...,
                *,
                inplace    : bool = ...,
                **kwargs   : Any,
                ) -> PrivSeries[T]: ...

    def replace(self,
                to_replace : Any  = None,
                value      : Any  = None,
                *,
                inplace    : bool = False,
                **kwargs   : Any,
                ) -> PrivSeries[T] | None:
        if (not is_realnum(to_replace)) or (not is_realnum(value)):
            # TODO: consider string and category dtype
            raise NotImplementedError

        if self.domain.type == "int64" and _np.isnan(value):
            new_domain = copy.copy(self.domain)
            new_domain.type = "Int64"

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
            self._value.replace(to_replace, value, inplace=inplace, **kwargs) # type: ignore[arg-type]
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data=self._value.replace(to_replace, value, inplace=inplace, **kwargs), domain=new_domain, distance=self.distance, parents=[self], preserve_row=True) # type: ignore[arg-type]

    @overload
    def dropna(self,
               *,
               inplace      : Literal[True],
               ignore_index : bool = ...,
               **kwargs     : Any,
               ) -> None: ...

    @overload
    def dropna(self,
               *,
               inplace      : bool = ...,
               ignore_index : bool = ...,
               **kwargs     : Any,
               ) -> PrivSeries[T]: ...

    def dropna(self,
               *,
               inplace      : bool = False,
               ignore_index : bool = False,
               **kwargs : Any,
               ) -> PrivSeries[T] | None:
        if ignore_index:
            raise DPError("`ignore_index` must be False. Index cannot be reindexed with positions.")

        if self.domain.type == "Int64":
            new_domain = copy.copy(self.domain)
            new_domain.type = "int64"
        else:
            new_domain = self.domain

        if inplace:
            self._value.dropna(inplace=inplace, **kwargs)
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data=self._value.dropna(inplace=inplace, **kwargs), domain=new_domain, distance=self.distance, parents=[self], preserve_row=True)

    @overload
    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : Literal[True],
             **kwargs : Any,
             ) -> None: ...

    @overload
    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : bool = ...,
             **kwargs : Any,
             ) -> PrivSeries[T]: ...

    def clip(self,
             lower    : realnum | None = None,
             upper    : realnum | None = None,
             *,
             inplace  : bool = False,
             **kwargs : Any,
             ) -> PrivSeries[T] | None:
        if not isinstance(self.domain, RealDomain):
            raise TypeError("Domain must be real numbers.")

        new_domain = copy.copy(self.domain)
        a, b = self.domain.range
        new_a = a if lower is None else lower if a is None else max(a, lower) # type: ignore[type-var]
        new_b = b if upper is None else upper if b is None else min(b, upper) # type: ignore[type-var]
        new_domain.range = (new_a, new_b)

        if inplace:
            self._value.clip(lower, upper, inplace=inplace, **kwargs) # type: ignore[arg-type]
            self._domain = new_domain
            return None
        else:
            return PrivSeries[T](data=self._value.clip(lower, upper, inplace=inplace, **kwargs), domain=new_domain, distance=self.distance, parents=[self], preserve_row=True) # type: ignore[arg-type]

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

        if self.domain.type in ["int64", "Int64"]:
            return SensitiveInt(s, new_distance, parents=[self])
        elif self.domain.type in ["float64", "Float64"]:
            return SensitiveFloat(s, new_distance, parents=[self])
        else:
            raise ValueError

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
                     normalize : bool                               = False,
                     sort      : bool                               = True,
                     ascending : bool                               = False,
                     bins      : int | None                         = None,
                     dropna    : bool                               = True,
                     values    : list[Any] | _pd.Series[Any] | None = None, # extra argument for privjail
                     ) -> SensitiveSeries[int]:
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

        if not dropna and not any(_np.isnan(values)):
            # TODO: consider handling for pd.NA
            warnings.warn("Counts for NaN will be dropped from the result because NaN is not included in `values`", UserWarning)

        counts = self._value.value_counts(normalize, sort, ascending, bins, dropna)

        # Select only the specified values and fill non-existent counts with 0
        counts = counts.reindex(values).fillna(0).astype(int)

        distances = self.distance.create_exclusive_distances(counts.size)

        prisoner_dummy = Prisoner(0, self.distance, parents=[self], children_type="exclusive")

        priv_counts = SensitiveSeries[int](index=counts.index, dtype="object")
        for i, idx in enumerate(counts.index):
            priv_counts.loc[idx] = SensitiveInt(counts.loc[idx], distance=distances[i], parents=[prisoner_dummy])

        return priv_counts

class SensitiveDataFrame(_pd.DataFrame):
    """Sensitive DataFrame.

    Each value in this dataframe object is considered a sensitive value.
    The numbers of rows and columns are not sensitive.
    This is typically created by counting queries like `pandas.crosstab()` and `pandas.pivot_table()`.
    """
    pass

# to avoid TypeError: type 'Series' is not subscriptable
# class SensitiveSeries(_pd.Series[T]):
class SensitiveSeries(Generic[T], _pd.Series): # type: ignore[type-arg]
    """Sensitive Series.

    Each value in this series object is considered a sensitive value.
    The numbers of values are not sensitive.
    This is typically created by counting queries like `PrivSeries.value_counts()`.
    """

    def max(self, *args: Any, **kwargs: Any) -> SensitiveInt | SensitiveFloat:
        # TODO: args?
        return smax(self)

    def min(self, *args: Any, **kwargs: Any) -> SensitiveInt | SensitiveFloat:
        # TODO: args?
        return smin(self)

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

def crosstab(index        : PrivSeries[Any] | list[PrivSeries[Any]],
             columns      : PrivSeries[Any] | list[PrivSeries[Any]],
             values       : PrivSeries[Any] | None                          = None,
             rownames     : list[str] | None                                = None,
             colnames     : list[str] | None                                = None,
             rowvalues    : list[Any] | _pd.Series[Any] | None              = None, # extra argument for privjail
             colvalues    : list[Any] | _pd.Series[Any] | None              = None, # extra argument for privjail
             *,
             aggfunc      : Any                                             = None,
             margins      : bool                                            = False,
             margins_name : str                                             = "All",
             dropna       : bool                                            = True,
             normalize    : bool | Literal["all", "index", "columns", 0, 1] = False,
             ) -> SensitiveDataFrame:
    if normalize is not False:
        # TODO: what is the sensitivity?
        raise NotImplementedError

    if values is not None or aggfunc is not None:
        # TODO: hard to accept arbitrary user functions
        raise NotImplementedError

    if margins:
        # Sensitivity must be managed separately for value counts and margins
        raise DPError("`margins=True` is not supported. Please manually calculate margins after adding noise.")

    if rowvalues is None or colvalues is None:
        # TODO: track the value domain to automatically determine the output dimension
        raise DPError("Please provide the `rowvalues`/`colvalues` arguments to prevent privacy leakage.")

    if not dropna and (not any(_np.isnan(rowvalues)) or not any(_np.isnan(colvalues))):
        # TODO: consider handling for pd.NA
        warnings.warn("Counts for NaN will be dropped from the result because NaN is not included in `rowvalues`/`colvalues`", UserWarning)

    if isinstance(index, list):
        if len(index) == 0:
            raise ValueError("Empty list is passed to `index`.")

        # TODO: support list of series (need to accept multiindex for rowvalues/colvalues)
        raise NotImplementedError
    else:
        index = [index]

    if isinstance(columns, list):
        if len(columns) == 0:
            raise ValueError("Empty list is passed to `columns`.")

        # TODO: support list of series (need to accept multiindex for rowvalues/colvalues)
        raise NotImplementedError
    else:
        columns = [columns]

    if not all(index[0]._ptag == x._ptag for x in index + columns):
        # TODO: reconsider whether this should be disallowed, as length difference does not cause an error
        raise DPError("Series in `index` and `columns` must have the same length.")

    counts = _pd.crosstab([x._value for x in index], [x._value for x in columns],
                          values=None, rownames=rownames, colnames=colnames,
                          aggfunc=None, margins=False, margins_name=margins_name,
                          dropna=dropna, normalize=False)

    # Select only the specified values and fill non-existent counts with 0
    counts = counts.reindex(rowvalues, axis="index").reindex(colvalues, axis="columns").fillna(0).astype(int)

    distances = index[0].distance.create_exclusive_distances(counts.size)

    prisoner_dummy = Prisoner(0, index[0].distance, parents=index + columns, children_type="exclusive")

    priv_counts = SensitiveDataFrame(index=counts.index, columns=counts.columns)
    i = 0
    for idx in counts.index:
        for col in counts.columns:
            priv_counts.loc[idx, col] = SensitiveInt(counts.loc[idx, col], distance=distances[i], parents=[prisoner_dummy]) # type: ignore
            i += 1

    return priv_counts

def cut(x        : PrivSeries[Any],
        bins     : list[int] | list[float],
        *args    : Any,
        **kwargs : Any,
        ) -> PrivSeries[Any]:
    if not isinstance(x, PrivSeries):
        raise DPError("`x` must be a PrivSeries.")

    if not isinstance(bins, list):
        raise DPError("`bins` must be a list.")

    ser = _pd.cut(x._value, bins, *args, **kwargs)

    new_domain = CategoryDomain(list(ser.dtype.categories))

    return PrivSeries[Any](ser, domain=new_domain, distance=x.distance, parents=[x], preserve_row=True)
