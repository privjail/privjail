from __future__ import annotations
from typing import overload, TypeVar, Any, Literal, Iterator, Generic
import warnings

import numpy as _np
import pandas as _pd
from pandas.api.types import is_list_like
# FIXME: pandas.core.common API is not public
from pandas.core.common import is_bool_indexer

from .util import DPError
from .prisoner import Prisoner, SensitiveInt
from .distance import Distance

T = TypeVar("T")

@overload
def unwrap_prisoner(x: Prisoner[T]) -> T: ...
@overload
def unwrap_prisoner(x: T) -> T: ...

def unwrap_prisoner(x: Prisoner[T] | T) -> T:
    if isinstance(x, Prisoner):
        return x._value
    else:
        return x

def is_2d_array(x: Any) -> bool:
    return (
        (isinstance(x, _np.ndarray) and x.ndim >= 2) or
        (isinstance(x, list) and all(isinstance(i, list) for i in x))
    )

class PrivDataFrame(Prisoner[_pd.DataFrame]):
    """Private DataFrame.

    Each row in this dataframe object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of rows is treated as a sensitive value.
    """

    def __init__(self,
                 data         : Any,
                 distance     : Distance,
                 index        : Any                 = None,
                 columns      : Any                 = None,
                 dtype        : Any                 = None,
                 copy         : bool                = False,
                 *,
                 parents      : list[Prisoner[Any]] = [],
                 root_name    : str | None          = None,
                 preserve_row : bool | None         = None,
                 ):
        if len(parents) == 0:
            preserve_row = False
        elif preserve_row is None:
            raise ValueError("preserve_row is required when parents are specified.")

        data = _pd.DataFrame(data, index, columns, dtype, copy)
        if preserve_row:
            super().__init__(value=data, distance=distance, parents=parents, root_name=root_name, tag_type="inherit")
        else:
            super().__init__(value=data, distance=distance, parents=parents, root_name=root_name, tag_type="renew")

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(df) is not supported. Use df.shape[0] instead.")

    @overload
    def __getitem__(self, key: PrivDataFrame | PrivSeries[bool] | list[str]) -> PrivDataFrame: ...
    @overload
    def __getitem__(self, key: str) -> PrivSeries[Any]: ...

    def __getitem__(self, key: PrivDataFrame | PrivSeries[bool] | list[str] | str) -> PrivDataFrame | PrivSeries[Any]:
        if isinstance(key, slice):
            raise DPError("df[slice] cannot be accepted because len(df) can be leaked depending on len(slice).")

        if isinstance(key, Prisoner) and not is_bool_indexer(key._value):
            raise DPError("df[key] is not allowed for sensitive keys other than boolean vectors.")

        if not isinstance(key, Prisoner) and is_bool_indexer(key):
            raise DPError("df[bool_vec] cannot be accepted because len(df) can be leaked depending on len(bool_vec).")

        if isinstance(key, str):
            # TODO: consider duplicated column names
            return PrivSeries[Any](data=self._value.__getitem__(key), distance=self.distance, parents=[self], preserve_row=True)

        elif isinstance(key, list) and all(isinstance(x, str) for x in key):
            return PrivDataFrame(data=self._value.__getitem__(key), distance=self.distance, parents=[self], preserve_row=True)

        elif isinstance(key, Prisoner) and is_bool_indexer(key._value):
            if not self.has_same_tag(key):
                raise DPError("df[bool_vec] cannot be accepted when df and bool_vec are not row-preserved.")

            return PrivDataFrame(data=self._value.__getitem__(key._value), distance=self.distance, parents=[self, key], preserve_row=False)

        else:
            raise ValueError

    def __setitem__(self, key: PrivDataFrame | PrivSeries[bool] | list[str] | str, value: Any) -> None:
        if isinstance(key, slice):
            raise DPError("df[slice] cannot be accepted because len(df) can be leaked depending on len(slice).")

        if isinstance(key, Prisoner) and not is_bool_indexer(key._value):
            raise DPError("df[key] is not allowed for sensitive keys other than boolean vectors.")

        if not isinstance(key, Prisoner) and is_bool_indexer(key):
            raise DPError("df[bool_vec] cannot be accepted because len(df) can be leaked depending on len(bool_vec).")

        if isinstance(value, Prisoner) and not isinstance(value, (PrivDataFrame, PrivSeries)):
            raise DPError("Sensitive values (other than PrivDataFrame and PrivSeries) cannot be assigned to dataframe.")

        if isinstance(key, Prisoner) and is_bool_indexer(key._value) and isinstance(value, PrivSeries):
            raise DPError("Private Series cannot be assigned to filtered rows because the series is treated as columns here.")

        if is_2d_array(value):
            raise DPError("2D array cannot be assigned because len(df) can be leaked.")

        if not isinstance(key, Prisoner) and not is_bool_indexer(key) and \
                not isinstance(value, Prisoner) and is_list_like(value) and not isinstance(value, (_pd.DataFrame, _pd.Series)):
            raise DPError("List-like values (other than DataFrame and Series) cannot be assigned because len(df) can be leaked.")

        if isinstance(key, Prisoner) and is_bool_indexer(key._value) and not self.has_same_tag(key):
            raise DPError("df[bool_vec] cannot be accepted when len(df) and len(bool_vec) can be different.")

        self._value[unwrap_prisoner(key)] = unwrap_prisoner(value)

    def __eq__(self, other: Any) -> PrivDataFrame: # type: ignore[override]
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value == other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value == other, distance=self.distance, parents=[self], preserve_row=True)

    def __ne__(self, other: Any) -> PrivDataFrame: # type: ignore[override]
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value != other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value != other, distance=self.distance, parents=[self], preserve_row=True)

    def __lt__(self, other: Any) -> PrivDataFrame:
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")
            return PrivDataFrame(data=self._value < other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value < other, distance=self.distance, parents=[self], preserve_row=True)

    def __le__(self, other: Any) -> PrivDataFrame:
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value <= other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value <= other, distance=self.distance, parents=[self], preserve_row=True)

    def __gt__(self, other: Any) -> PrivDataFrame:
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value > other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value > other, distance=self.distance, parents=[self], preserve_row=True)

    def __ge__(self, other: Any) -> PrivDataFrame:
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value >= other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value >= other, distance=self.distance, parents=[self], preserve_row=True)

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
        if inplace:
            self._value.replace(to_replace, value, inplace=inplace, **kwargs)
            return None
        else:
            return PrivDataFrame(data=self._value.replace(to_replace, value, inplace=inplace, **kwargs), distance=self.distance, parents=[self], preserve_row=True)

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

        if inplace:
            self._value.dropna(inplace=inplace, **kwargs)
            return None
        else:
            return PrivDataFrame(data=self._value.dropna(inplace=inplace, **kwargs), distance=self.distance, parents=[self], preserve_row=True)

    def groupby(self,
                by         : str | list[str], # TODO: support more
                level      : Any                                = None,
                as_index   : bool                               = True,
                sort       : bool                               = True,
                group_keys : bool                               = True,
                observed   : bool                               = True,
                dropna     : bool                               = True,
                keys       : list[Any] | _pd.Series[Any] | None = None, # extra argument for pripri
                ) -> PrivDataFrameGroupBy:
        if keys is None:
            # TODO: track the value domain to automatically determine the output dimension
            raise DPError("Please provide the `keys` argument to prevent privacy leakage.")

        # TODO: handling for arguments

        groupby_obj = self._value.groupby(by, level=level, as_index=as_index, sort=sort,
                                          group_keys=group_keys, observed=observed, dropna=dropna)

        # exclude groups not included in the specified `keys`
        groups = {key: df for key, df in groupby_obj if key in keys}

        # include empty groups for absent `keys` and sort by `keys`
        columns = self._value.columns
        dtypes = self._value.dtypes
        groups = {key: groups.get(key, _pd.DataFrame({c: _pd.Series(dtype=d) for c, d in zip(columns, dtypes)})) for key in keys}

        # create new child distance variables to express exclusiveness
        distances = self.distance.create_exclusive_distances(len(groups))

        # create a dummy prisoner to track exclusive provenance
        prisoner_dummy = Prisoner(0, self.distance, parents=[self], tag_type="renew", children_type="exclusive")

        # wrap each group by PrivDataFrame
        priv_groups = {key: PrivDataFrame(df, distance=d, parents=[prisoner_dummy], preserve_row=False) for (key, df), d in zip(groups.items(), distances)}

        return PrivDataFrameGroupBy(priv_groups)

class PrivDataFrameGroupBy:
    def __init__(self, groups: dict[Any, PrivDataFrame]):
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

    def __init__(self,
                 data         : Any,
                 distance     : Distance,
                 index        : Any                 = None,
                 *,
                 parents      : list[Prisoner[Any]] = [],
                 root_name    : str | None          = None,
                 preserve_row : bool | None         = None,
                 **kwargs    : Any,
                 ):
        if len(parents) == 0:
            preserve_row = False
        elif preserve_row is None:
            raise ValueError("preserve_row is required when parents are specified.")

        data = _pd.Series(data, index, **kwargs)
        if preserve_row:
            super().__init__(value=data, distance=distance, parents=parents, root_name=root_name, tag_type="inherit")
        else:
            super().__init__(value=data, distance=distance, parents=parents, root_name=root_name, tag_type="renew")

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(ser) is not supported. Use ser.shape[0] or ser.size instead.")

    def __getitem__(self, key: PrivSeries[bool]) -> PrivSeries[T]:
        if isinstance(key, slice):
            raise DPError("ser[slice] cannot be accepted because len(ser) can be leaked depending on len(slice).")

        if not isinstance(key, Prisoner):
            raise DPError("ser[bool_vec] cannot be accepted because len(ser) can be leaked depending on len(bool_vec).")

        if not is_bool_indexer(key._value):
            raise DPError("ser[key] is not allowed for sensitive keys other than boolean vectors.")

        if not self.has_same_tag(key):
            raise DPError("ser[bool_vec] cannot be accepted when ser and bool_vec are not row-preserved.")

        return PrivSeries[T](data=self._value.__getitem__(unwrap_prisoner(key)), distance=self.distance, parents=[self], preserve_row=False)

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice):
            raise DPError("ser[slice] cannot be accepted because len(ser) can be leaked depending on len(slice).")

        if not isinstance(key, Prisoner):
            raise DPError("ser[bool_vec] cannot be accepted because len(ser) can be leaked depending on len(bool_vec).")

        if not is_bool_indexer(key._value):
            raise DPError("ser[key] is not allowed for sensitive keys other than boolean vectors.")

        if isinstance(value, Prisoner) and not isinstance(value, (PrivDataFrame, PrivSeries)):
            raise DPError("Sensitive values (other than PrivDataFrame and PrivSeries) cannot be assigned to dataframe.")

        if isinstance(value, PrivSeries):
            raise DPError("Private Series cannot be assigned to filtered rows because the series is treated as columns here.")

        if is_2d_array(value):
            raise DPError("2D array cannot be assigned because len(df) can be leaked.")

        if isinstance(key, Prisoner) and is_bool_indexer(key._value) and not self.has_same_tag(key):
            raise DPError("df[bool_vec] cannot be accepted when len(df) and len(bool_vec) can be different.")

        self._value[unwrap_prisoner(key)] = unwrap_prisoner(value)

    def __eq__(self, other: Any) -> PrivSeries[bool]: # type: ignore[override]
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value == other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value == other, distance=self.distance, parents=[self], preserve_row=True)

    def __ne__(self, other: Any) -> PrivSeries[bool]: # type: ignore[override]
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value != other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value != other, distance=self.distance, parents=[self], preserve_row=True)

    def __lt__(self, other: Any) -> PrivSeries[bool]:
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value < other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value < other, distance=self.distance, parents=[self], preserve_row=True)

    def __le__(self, other: Any) -> PrivSeries[bool]:
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value <= other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value <= other, distance=self.distance, parents=[self], preserve_row=True)

    def __gt__(self, other: Any) -> PrivSeries[bool]:
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value > other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value > other, distance=self.distance, parents=[self], preserve_row=True)

    def __ge__(self, other: Any) -> PrivSeries[bool]:
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries[bool](data=self._value >= other._value, distance=self.distance, parents=[self, other], preserve_row=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries[bool](data=self._value >= other, distance=self.distance, parents=[self], preserve_row=True)

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
        if inplace:
            self._value.replace(to_replace, value, inplace=inplace, **kwargs)
            return None
        else:
            return PrivSeries[T](data=self._value.replace(to_replace, value, inplace=inplace, **kwargs), distance=self.distance, parents=[self], preserve_row=True)

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

        if inplace:
            self._value.dropna(inplace=inplace, **kwargs)
            return None
        else:
            return PrivSeries[T](data=self._value.dropna(inplace=inplace, **kwargs), distance=self.distance, parents=[self], preserve_row=True)

    def value_counts(self,
                     normalize : bool                               = False,
                     sort      : bool                               = True,
                     ascending : bool                               = False,
                     bins      : int | None                         = None,
                     dropna    : bool                               = True,
                     values    : list[Any] | _pd.Series[Any] | None = None, # extra argument for pripri
                     ) -> SensitiveSeries[int]:
        if normalize:
            # TODO: what is the sensitivity?
            raise NotImplementedError

        if bins is not None:
            # TODO: support continuous values
            raise NotImplementedError

        if sort:
            raise DPError("The `sort` argument must be False.")

        if values is None:
            # TODO: track the value domain to automatically determine the output dimension
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

        prisoner_dummy = Prisoner(0, self.distance, parents=[self], tag_type="renew", children_type="exclusive")

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
    pass

def read_csv(filepath: str, **kwargs: Any) -> PrivDataFrame:
    return PrivDataFrame(data=_pd.read_csv(filepath, **kwargs), distance=Distance(1), root_name=filepath)

def crosstab(index        : PrivSeries[Any] | list[PrivSeries[Any]],
             columns      : PrivSeries[Any] | list[PrivSeries[Any]],
             values       : PrivSeries[Any] | None                          = None,
             rownames     : list[str] | None                                = None,
             colnames     : list[str] | None                                = None,
             rowvalues    : list[Any] | _pd.Series[Any] | None              = None, # extra argument for pripri
             colvalues    : list[Any] | _pd.Series[Any] | None              = None, # extra argument for pripri
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

    if not all(index[0].has_same_tag(x) for x in index + columns):
        # TODO: reconsider whether this should be disallowed, as length difference does not cause an error
        raise DPError("Series in `index` and `columns` must have the same length.")

    counts = _pd.crosstab([x._value for x in index], [x._value for x in columns],
                          values=None, rownames=rownames, colnames=colnames,
                          aggfunc=None, margins=False, margins_name=margins_name,
                          dropna=dropna, normalize=False)

    # Select only the specified values and fill non-existent counts with 0
    counts = counts.reindex(rowvalues, axis="index").reindex(colvalues, axis="columns").fillna(0).astype(int)

    distances = index[0].distance.create_exclusive_distances(counts.size)

    prisoner_dummy = Prisoner(0, index[0].distance, parents=list(index + columns), tag_type="renew", children_type="exclusive")

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

    return PrivSeries[Any](_pd.cut(x._value, bins, *args, **kwargs), distance=x.distance, parents=[x], preserve_row=True)
