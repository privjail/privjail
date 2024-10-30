from .util import DPError
from .prisoner import Prisoner
import warnings
import numpy as _np
import pandas as _pd
from pandas.api.types import is_list_like
# FIXME: pandas.core.common API is not public
from pandas.core.common import is_bool_indexer

def unwrap_prisoner(x):
    if isinstance(x, Prisoner):
        return x._value
    else:
        return x

def is_2d_array(x):
    return (
        (isinstance(x, _np.ndarray) and x.ndim >= 2) or
        (isinstance(x, list) and all(isinstance(i, list) for i in x))
    )

class PrivDataFrame(Prisoner):
    """Private DataFrame.

    Each row in this dataframe object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of rows is treated as a sensitive value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(value=_pd.DataFrame(*args, **kwargs), sensitivity=-1)

    def __len__(self):
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(df) is not supported. Use df.shape[0] instead.")

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise DPError("df[slice] cannot be accepted because len(df) can be leaked depending on len(slice).")

        if isinstance(key, Prisoner) and not is_bool_indexer(key._value):
            raise DPError("df[key] is not allowed for sensitive keys other than boolean vectors.")

        if not isinstance(key, Prisoner) and is_bool_indexer(key):
            raise DPError("df[bool_vec] cannot be accepted because len(df) can be leaked depending on len(bool_vec).")

        data = self._value.__getitem__(unwrap_prisoner(key))
        if isinstance(data, _pd.DataFrame):
            return PrivDataFrame(data)
        elif isinstance(data, _pd.Series):
            return PrivSeries(data)
        else:
            raise RuntimeError

    def __setitem__(self, key, value):
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

        self._value[unwrap_prisoner(key)] = unwrap_prisoner(value)

    def __eq__(self, other):
        if isinstance(other, PrivDataFrame):
            # FIXME: length leakage problem
            return PrivDataFrame(data=self._value == other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return PrivDataFrame(data=self._value == other)

    def __ne__(self, other):
        if isinstance(other, PrivDataFrame):
            # FIXME: length leakage problem
            return PrivDataFrame(data=self._value != other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return PrivDataFrame(data=self._value != other)

    def __lt__(self, other):
        if isinstance(other, PrivDataFrame):
            # FIXME: length leakage problem
            return PrivDataFrame(data=self._value < other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return PrivDataFrame(data=self._value < other)

    def __le__(self, other):
        if isinstance(other, PrivDataFrame):
            # FIXME: length leakage problem
            return PrivDataFrame(data=self._value <= other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return PrivDataFrame(data=self._value <= other)

    def __gt__(self, other):
        if isinstance(other, PrivDataFrame):
            # FIXME: length leakage problem
            return PrivDataFrame(data=self._value > other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return PrivDataFrame(data=self._value > other)

    def __ge__(self, other):
        if isinstance(other, PrivDataFrame):
            # FIXME: length leakage problem
            return PrivDataFrame(data=self._value >= other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return PrivDataFrame(data=self._value >= other)

    @property
    def shape(self):
        nrows = Prisoner(value=self._value.shape[0], sensitivity=1)
        ncols = self._value.shape[1]
        return (nrows, ncols)

    @property
    def size(self):
        return Prisoner(value=self._value.size, sensitivity=len(self._value.columns))

    @property
    def columns(self):
        return self._value.columns

    @columns.setter
    def columns(self, value):
        self._value.columns = value

    @property
    def dtypes(self):
        return self._value.dtypes

    def reset_index(self, *args, **kwargs):
        if kwargs.get("inplace", False):
            self._value.reset_index(*args, **kwargs)
            return None
        else:
            return PrivDataFrame(data=self._value.reset_index(*args, **kwargs))

class PrivSeries(Prisoner):
    """Private Series.

    Each value in this series object should have a one-to-one relationship with an individual (event-/row-/item-level DP).
    Therefore, the number of values is treated as a sensitive value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(value=_pd.Series(*args, **kwargs), sensitivity=-1)

    def __len__(self):
        # We cannot return Prisoner() here because len() must be an integer value
        raise DPError("len(ser) is not supported. Use ser.shape[0] or ser.size instead.")

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise DPError("ser[slice] cannot be accepted because len(ser) can be leaked depending on len(slice).")

        if not isinstance(key, Prisoner):
            raise DPError("ser[bool_vec] cannot be accepted because len(ser) can be leaked depending on len(bool_vec).")

        if not is_bool_indexer(key._value):
            raise DPError("ser[key] is not allowed for sensitive keys other than boolean vectors.")

        return PrivSeries(data=self._value.__getitem__(unwrap_prisoner(key)))

    def __setitem__(self, key, value):
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

        self._value[unwrap_prisoner(key)] = unwrap_prisoner(value)

    def __eq__(self, other):
        if isinstance(other, PrivSeries):
            # FIXME: length leakage problem
            return PrivSeries(data=self._value == other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return PrivSeries(data=self._value == other)

    def __ne__(self, other):
        if isinstance(other, PrivSeries):
            # FIXME: length leakage problem
            return PrivSeries(data=self._value != other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return PrivSeries(data=self._value != other)

    def __lt__(self, other):
        if isinstance(other, PrivSeries):
            # FIXME: length leakage problem
            return PrivSeries(data=self._value < other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return PrivSeries(data=self._value < other)

    def __le__(self, other):
        if isinstance(other, PrivSeries):
            # FIXME: length leakage problem
            return PrivSeries(data=self._value <= other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return PrivSeries(data=self._value <= other)

    def __gt__(self, other):
        if isinstance(other, PrivSeries):
            # FIXME: length leakage problem
            return PrivSeries(data=self._value > other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return PrivSeries(data=self._value > other)

    def __ge__(self, other):
        if isinstance(other, PrivSeries):
            # FIXME: length leakage problem
            return PrivSeries(data=self._value >= other._value)
        elif isinstance(other, Prisoner):
            raise DPError("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise DPError("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return PrivSeries(data=self._value >= other)

    @property
    def shape(self):
        nrows = Prisoner(value=self._value.shape[0], sensitivity=1)
        return (nrows,)

    @property
    def size(self):
        return Prisoner(value=self._value.size, sensitivity=1)

    @property
    def dtypes(self):
        return self._value.dtypes

    def reset_index(self, *args, **kwargs):
        if kwargs.get("inplace", False):
            self._value.reset_index(*args, **kwargs)
            return None
        elif kwargs.get("drop", False):
            return PrivSeries(data=self._value.reset_index(*args, **kwargs))
        else:
            return PrivDataFrame(data=self._value.reset_index(*args, **kwargs))

    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True,
                     values=None): # extra arguments for pripri
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
        counts = counts.reindex(values, axis="index").fillna(0).astype(int)

        return SensitiveSeries(data=counts, sensitivity=1)

class SensitiveDataFrame(Prisoner):
    """Sensitive DataFrame.

    Each value in this dataframe object is considered a sensitive value.
    The numbers of rows and columns are not sensitive.
    This is typically created by counting queries like `pandas.crosstab()` and `pandas.pivot_table()`.
    """

    def __init__(self, sensitivity, *args, **kwargs):
        if "sensitivity" not in kwargs:
            raise ValueError("sensitivity must be provided as a keyword argument.")

        sensitivity = kwargs["sensitivity"]
        kwargs.pop("sensitivity")
        super().__init__(value=_pd.DataFrame(*args, **kwargs), sensitivity=sensitivity)

    def __len__(self):
        return len(self._value)

    @property
    def shape(self):
        return self._value.shape

    @property
    def size(self):
        return self._value.size

    @property
    def index(self):
        return self._value.index

    @index.setter
    def index(self, value):
        self._value.index = value

    @property
    def columns(self):
        return self._value.columns

    @columns.setter
    def columns(self, value):
        self._value.columns = value

    @property
    def dtypes(self):
        return self._value.dtypes

    def __getitem__(self, key):
        if isinstance(key, Prisoner):
            raise DPError("Sensitive values cannot be accepted as keys for sensitive dataframe.")

        data = self._value.__getitem__(key)
        if isinstance(data, _pd.DataFrame):
            return SensitiveDataFrame(data=data, sensitivity=self.sensitivity)
        elif isinstance(data, _pd.Series):
            return SensitiveSeries(data=data, sensitivity=self.sensitivity)
        else:
            return RuntimeError

    def __setitem__(self, key, value):
        raise DPError("Assignment to sensitive datais not allowed.")

    def __eq__(self, other):
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __ne__(self, other):
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __lt__(self, other):
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __le__(self, other):
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __gt__(self, other):
        raise DPError("Comparison against sensitive dataframe is not allowed.")

    def __ge__(self, other):
        raise DPError("Comparison against sensitive dataframe is not allowed.")

class SensitiveSeries(Prisoner):
    """Sensitive Series.

    Each value in this series object is considered a sensitive value.
    The numbers of values are not sensitive.
    This is typically created by counting queries like `PrivSeries.value_counts()`.
    """

    def __init__(self, *args, **kwargs):
        if "sensitivity" not in kwargs:
            raise ValueError("sensitivity must be provided as a keyword argument.")

        sensitivity = kwargs["sensitivity"]
        kwargs.pop("sensitivity")
        super().__init__(value=_pd.Series(*args, **kwargs), sensitivity=sensitivity)

    def __len__(self):
        return len(self._value)

    @property
    def shape(self):
        return self._value.shape

    @property
    def size(self):
        return self._value.size

    @property
    def index(self):
        return self._value.index

    @index.setter
    def index(self, value):
        self._value.index = value

    @property
    def dtypes(self):
        return self._value.dtypes

    def __getitem__(self, key):
        if isinstance(key, Prisoner):
            raise DPError("Sensitive values cannot be accepted as keys for sensitive series.")

        data = self._value.__getitem__(key)
        if isinstance(data, _pd.Series):
            return SensitiveSeries(data=data, sensitivity=self.sensitivity)
        else:
            return Prisoner(value=data, sensitivity=self.sensitivity)

    def __setitem__(self, key, value):
        raise DPError("Assignment to sensitive series is not allowed.")

    def __eq__(self, other):
        raise DPError("Comparison against sensitive series is not allowed.")

    def __ne__(self, other):
        raise DPError("Comparison against sensitive series is not allowed.")

    def __lt__(self, other):
        raise DPError("Comparison against sensitive series is not allowed.")

    def __le__(self, other):
        raise DPError("Comparison against sensitive series is not allowed.")

    def __gt__(self, other):
        raise DPError("Comparison against sensitive series is not allowed.")

    def __ge__(self, other):
        raise DPError("Comparison against sensitive series is not allowed.")

def read_csv(*args, **kwargs):
    return PrivDataFrame(data=_pd.read_csv(*args, **kwargs))
