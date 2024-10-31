from __future__ import annotations
from typing import overload, TypeVar, Any, Literal
import warnings

import numpy as _np
import pandas as _pd
from pandas.api.types import is_list_like
# FIXME: pandas.core.common API is not public
from pandas.core.common import is_bool_indexer

from .util import DPError
from .prisoner import Prisoner

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
                 data        : Any,
                 index       : Any                  = None,
                 columns     : Any                  = None,
                 dtype       : Any                  = None,
                 copy        : bool                 = False,
                 *,
                 parent      : Prisoner[Any] | None = None,
                 root_name   : str | None           = None,
                 inherit_len : bool | None          = None,
                 ):
        if parent is None:
            inherit_len = False
        elif inherit_len is None:
            raise ValueError("inherit_len is required when parent is specified.")

        data = _pd.DataFrame(data, index, columns, dtype, copy)
        super().__init__(value=data, sensitivity=-1, parent=parent, root_name=root_name, renew_tag=not inherit_len)

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(df) is not supported. Use df.shape[0] instead.")

    @overload
    def __getitem__(self, key: PrivDataFrame | PrivSeries | list[str]) -> PrivDataFrame: ...
    @overload
    def __getitem__(self, key: str) -> PrivSeries: ...

    def __getitem__(self, key: PrivDataFrame | PrivSeries | list[str] | str) -> PrivDataFrame | PrivSeries:
        if isinstance(key, slice):
            raise DPError("df[slice] cannot be accepted because len(df) can be leaked depending on len(slice).")

        if isinstance(key, Prisoner) and not is_bool_indexer(key._value):
            raise DPError("df[key] is not allowed for sensitive keys other than boolean vectors.")

        if not isinstance(key, Prisoner) and is_bool_indexer(key):
            raise DPError("df[bool_vec] cannot be accepted because len(df) can be leaked depending on len(bool_vec).")

        if isinstance(key, str):
            # TODO: consider duplicated column names
            return PrivSeries(data=self._value.__getitem__(key), parent=self, inherit_len=True)

        elif isinstance(key, list) and all(isinstance(x, str) for x in key):
            return PrivDataFrame(data=self._value.__getitem__(key), parent=self, inherit_len=True)

        elif isinstance(key, Prisoner) and is_bool_indexer(key._value):
            if not self.has_same_tag(key):
                raise DPError("df[bool_vec] cannot be accepted when len(df) and len(bool_vec) can be different.")

            return PrivDataFrame(data=self._value.__getitem__(key._value), parent=self, inherit_len=False)

        else:
            raise ValueError

    def __setitem__(self, key: PrivDataFrame | PrivSeries | list[str] | str, value: Any) -> None:
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

            return PrivDataFrame(data=self._value == other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value == other, parent=self, inherit_len=True)

    def __ne__(self, other: Any) -> PrivDataFrame: # type: ignore[override]
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value != other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value != other, parent=self, inherit_len=True)

    def __lt__(self, other: Any) -> PrivDataFrame:
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")
            return PrivDataFrame(data=self._value < other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value < other, parent=self, inherit_len=True)

    def __le__(self, other: Any) -> PrivDataFrame:
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value <= other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value <= other, parent=self, inherit_len=True)

    def __gt__(self, other: Any) -> PrivDataFrame:
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value > other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value > other, parent=self, inherit_len=True)

    def __ge__(self, other: Any) -> PrivDataFrame:
        if isinstance(other, PrivDataFrame):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive dataframes for comparison can be different.")

            return PrivDataFrame(data=self._value >= other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")

        else:
            return PrivDataFrame(data=self._value >= other, parent=self, inherit_len=True)

    @property
    def shape(self) -> tuple[Prisoner[int], int]:
        nrows = Prisoner(value=self._value.shape[0], sensitivity=1, parent=self)
        ncols = self._value.shape[1]
        return (nrows, ncols)

    @property
    def size(self) -> Prisoner[int]:
        return Prisoner(value=self._value.size, sensitivity=len(self._value.columns), parent=self)

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
    def reset_index(self,
                    level    : Any = ...,
                    *,
                    inplace  : Literal[True],
                    **kwargs : Any,
                    ) -> None: ...
    @overload
    def reset_index(self,
                    level    : Any  = ...,
                    *,
                    inplace  : bool = ...,
                    **kwargs : Any,
                    ) -> PrivDataFrame: ...

    def reset_index(self,
                    level    : Any  = None,
                    *,
                    inplace  : bool = False,
                    **kwargs : Any,
                    ) -> PrivDataFrame | None:
        if inplace:
            self._value.reset_index(level, inplace=inplace, **kwargs)
            return None
        else:
            return PrivDataFrame(data=self._value.reset_index(level, inplace=inplace, **kwargs), parent=self, inherit_len=True)

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

        groupby_obj = self._value.groupby(by, level=level, as_index=as_index, sort=sort,
                                          group_keys=group_keys, observed=observed, dropna=dropna)
        return PrivDataFrameGroupBy(groupby_obj, parent=self)

class PrivDataFrameGroupBy(Prisoner[_pd.core.groupby.DataFrameGroupBy]):
    groups : dict[Any, PrivDataFrame]

    def __init__(self,
                 groupby_obj : Any,
                 *,
                 parent      : Prisoner[Any],
                 ):
        # TODO: probably do not need to save groupby_obj
        super().__init__(value=groupby_obj, sensitivity=-1, parent=parent, children_type="exclusive")
        self.groups = {key: PrivDataFrame(df, parent=self, inherit_len=False) for key, df in groupby_obj}

    def __iter__(self):
        return iter(self.groups.items())

    def get_group(self, key):
        return self.groups[key]

class PrivSeries(Prisoner[_pd.Series]): # type: ignore[type-arg]
    """Private Series.

    Each value in this series object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of values is treated as a sensitive value.
    """

    def __init__(self,
                 data        : Any,
                 index       : Any                  = None,
                 *,
                 parent      : Prisoner[Any] | None = None,
                 root_name   : str | None           = None,
                 inherit_len : bool | None          = None,
                 **kwargs    : Any,
                 ):
        if parent is None:
            inherit_len = False
        elif inherit_len is None:
            raise ValueError("inherit_len is required when parent is specified.")

        data = _pd.Series(data, index, **kwargs)
        super().__init__(value=data, sensitivity=-1, parent=parent, root_name=root_name, renew_tag=not inherit_len)

    def __len__(self) -> int:
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(ser) is not supported. Use ser.shape[0] or ser.size instead.")

    def __getitem__(self, key: PrivSeries) -> PrivSeries:
        if isinstance(key, slice):
            raise DPError("ser[slice] cannot be accepted because len(ser) can be leaked depending on len(slice).")

        if not isinstance(key, Prisoner):
            raise DPError("ser[bool_vec] cannot be accepted because len(ser) can be leaked depending on len(bool_vec).")

        if not is_bool_indexer(key._value):
            raise DPError("ser[key] is not allowed for sensitive keys other than boolean vectors.")

        if not self.has_same_tag(key):
            raise DPError("ser[bool_vec] cannot be accepted when len(ser) and len(bool_vec) can be different.")

        return PrivSeries(data=self._value.__getitem__(unwrap_prisoner(key)), parent=self, inherit_len=False)

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

    def __eq__(self, other: Any) -> PrivSeries: # type: ignore[override]
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries(data=self._value == other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries(data=self._value == other, parent=self, inherit_len=True)

    def __ne__(self, other: Any) -> PrivSeries: # type: ignore[override]
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries(data=self._value != other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries(data=self._value != other, parent=self, inherit_len=True)

    def __lt__(self, other: Any) -> PrivSeries:
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries(data=self._value < other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries(data=self._value < other, parent=self, inherit_len=True)

    def __le__(self, other: Any) -> PrivSeries:
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries(data=self._value <= other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries(data=self._value <= other, parent=self, inherit_len=True)

    def __gt__(self, other: Any) -> PrivSeries:
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries(data=self._value > other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries(data=self._value > other, parent=self, inherit_len=True)

    def __ge__(self, other: Any) -> PrivSeries:
        if isinstance(other, PrivSeries):
            if not self.has_same_tag(other):
                raise DPError("Length of sensitive series for comparison can be different.")

            return PrivSeries(data=self._value >= other._value, parent=self, inherit_len=True)

        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")

        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")

        else:
            return PrivSeries(data=self._value >= other, parent=self, inherit_len=True)

    @property
    def shape(self) -> tuple[Prisoner[int]]:
        nrows = Prisoner(value=self._value.shape[0], sensitivity=1, parent=self)
        return (nrows,)

    @property
    def size(self) -> Prisoner[int]:
        return Prisoner(value=self._value.size, sensitivity=1, parent=self)

    @property
    def dtypes(self) -> Any:
        return self._value.dtypes

    @overload
    def reset_index(self,
                    level    : Any  = ...,
                    *,
                    drop     : bool = ...,
                    inplace  : Literal[True],
                    **kwargs : Any,
                    ) -> None: ...
    @overload
    def reset_index(self,
                    level    : Any = ...,
                    *,
                    drop     : Literal[True],
                    inplace  : Literal[False],
                    **kwargs : Any,
                    ) -> PrivSeries: ...
    @overload
    def reset_index(self,
                    level    : Any  = ...,
                    *,
                    drop     : bool = ...,
                    inplace  : bool = ...,
                    **kwargs : Any,
                    ) -> PrivDataFrame: ...

    def reset_index(self,
                    level    : Any  = None,
                    *,
                    drop     : bool = False,
                    inplace  : bool = False,
                    **kwargs : Any,
                    ) -> PrivDataFrame | PrivSeries | None:
        if inplace:
            self._value.reset_index(level, drop=drop, inplace=inplace, **kwargs)
            return None
        elif drop:
            return PrivSeries(data=self._value.reset_index(level, drop=drop, inplace=inplace, **kwargs), parent=self, inherit_len=True)
        else:
            return PrivDataFrame(data=self._value.reset_index(level, drop=drop, inplace=inplace, **kwargs), parent=self, inherit_len=True)

    def value_counts(self,
                     normalize : bool                               = False,
                     sort      : bool                               = True,
                     ascending : bool                               = False,
                     bins      : int | None                         = None,
                     dropna    : bool                               = True,
                     values    : list[Any] | _pd.Series[Any] | None = None, # extra argument for pripri
                     ) -> SensitiveSeries:
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

        return SensitiveSeries(data=counts, sensitivity=1, parent=self)

class SensitiveDataFrame(Prisoner[_pd.DataFrame]):
    """Sensitive DataFrame.

    Each value in this dataframe object is considered a sensitive value.
    The numbers of rows and columns are not sensitive.
    This is typically created by counting queries like `pandas.crosstab()` and `pandas.pivot_table()`.
    """

    def __init__(self,
                 data        : Any,
                 index       : Any                  = None,
                 columns     : Any                  = None,
                 dtype       : Any                  = None,
                 copy        : bool                 = False,
                 *,
                 sensitivity : float,
                 parent      : Prisoner[Any] | None = None,
                 root_name   : str | None           = None,
                 ):
        data = _pd.DataFrame(data, index, columns, dtype, copy)
        super().__init__(value=data, sensitivity=sensitivity, parent=parent, root_name=root_name)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def shape(self) -> tuple[int, int]:
        return self._value.shape

    @property
    def size(self) -> int:
        return self._value.size

    @property
    def index(self) -> _pd.Index[Any]:
        return self._value.index

    @index.setter
    def index(self, value: _pd.Index[Any]) -> None:
        self._value.index = value

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
    def __getitem__(self, key: _pd.DataFrame | _pd.Series[Any] | list[Any]) -> SensitiveDataFrame: ...
    @overload
    def __getitem__(self, key: Any) -> SensitiveSeries: ...

    def __getitem__(self, key: Any) -> SensitiveDataFrame | SensitiveSeries:
        if isinstance(key, Prisoner):
            raise DPError("Sensitive values cannot be accepted as keys for sensitive dataframe.")

        data = self._value.__getitem__(key)
        if isinstance(data, _pd.DataFrame):
            return SensitiveDataFrame(data=data, sensitivity=self.sensitivity, parent=self)
        elif isinstance(data, _pd.Series):
            return SensitiveSeries(data=data, sensitivity=self.sensitivity, parent=self)
        else:
            raise RuntimeError

    def __setitem__(self, key: Any, value: Any) -> None:
        raise DPError("Assignment to sensitive datais not allowed.")

    def __eq__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __ne__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __lt__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __le__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __gt__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __ge__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive dataframe is not allowed.")

class SensitiveSeries(Prisoner[_pd.Series]): # type: ignore[type-arg]
    """Sensitive Series.

    Each value in this series object is considered a sensitive value.
    The numbers of values are not sensitive.
    This is typically created by counting queries like `PrivSeries.value_counts()`.
    """

    def __init__(self,
                 data        : Any,
                 index       : Any                  = None,
                 *,
                 sensitivity : float,
                 parent      : Prisoner[Any] | None = None,
                 root_name   : str | None           = None,
                 **kwargs    : Any,
                 ):
        data = _pd.Series(data, index, **kwargs)
        super().__init__(value=data, sensitivity=sensitivity, parent=parent, root_name=root_name)

    def __len__(self) -> int:
        return len(self._value)

    @property
    def shape(self) -> tuple[int]:
        return self._value.shape

    @property
    def size(self) -> int:
        return self._value.size

    @property
    def index(self) -> _pd.Index[Any]:
        return self._value.index

    @index.setter
    def index(self, value: _pd.Index[Any]) -> None:
        self._value.index = value

    @property
    def dtypes(self) -> Any:
        return self._value.dtypes

    @overload
    def __getitem__(self, key: _pd.Series[Any] | list[Any] | slice) -> SensitiveSeries: ...
    @overload
    def __getitem__(self, key: Any) -> Prisoner[Any]: ...

    def __getitem__(self, key: Any) -> SensitiveSeries | Prisoner[Any]:
        if isinstance(key, Prisoner):
            raise DPError("Sensitive values cannot be accepted as keys for sensitive series.")

        data = self._value.__getitem__(key)
        if isinstance(data, _pd.Series):
            return SensitiveSeries(data=data, sensitivity=self.sensitivity, parent=self)
        else:
            return Prisoner(value=data, sensitivity=self.sensitivity, parent=self)

    def __setitem__(self, key: Any, value: Any) -> None:
        raise DPError("Assignment to sensitive series is not allowed.")

    def __eq__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive series is not allowed.")

    def __ne__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive series is not allowed.")

    def __lt__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive series is not allowed.")

    def __le__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive series is not allowed.")

    def __gt__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive series is not allowed.")

    def __ge__(self, other: Any) -> Any:
        raise DPError("Comparison against sensitive series is not allowed.")

def read_csv(filepath: str, **kwargs: Any) -> PrivDataFrame:
    return PrivDataFrame(data=_pd.read_csv(filepath, **kwargs), root_name=filepath, inherit_len=False)

def crosstab(index        : PrivSeries | list[PrivSeries],
             columns      : PrivSeries | list[PrivSeries],
             values       : PrivSeries | None                               = None,
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

    # TODO: is it ok to set one of the series as the parent?
    return SensitiveDataFrame(data=counts, sensitivity=1, parent=index[0])
