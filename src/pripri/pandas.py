from .prisoner import Prisoner
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

class DataFrame(Prisoner):
    def __init__(self, *args, **kwargs):
        super().__init__(value=_pd.DataFrame(*args, **kwargs), sensitivity=-1)

    def __len__(self):
        # We cannot return Prisoner() here because len() must be an integer value
        raise Exception("len(df) is not supported. Use df.shape[0] instead.")

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise Exception("df[slice] cannot be accepted because len(df) can be leaked depending on len(slice).")

        if isinstance(key, Prisoner) and not is_bool_indexer(key._value):
            raise Exception("df[key] is not allowed for sensitive keys other than boolean vectors.")

        if not isinstance(key, Prisoner) and is_bool_indexer(key):
            raise Exception("df[bool_vec] cannot be accepted because len(df) can be leaked depending on len(bool_vec).")

        data = self._value.__getitem__(unwrap_prisoner(key))
        if isinstance(data, _pd.DataFrame):
            return DataFrame(data)
        elif isinstance(data, _pd.Series):
            return Series(data)
        else:
            raise RuntimeError

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            raise Exception("df[slice] cannot be accepted because len(df) can be leaked depending on len(slice).")

        if isinstance(key, Prisoner) and not is_bool_indexer(key._value):
            raise Exception("df[key] is not allowed for sensitive keys other than boolean vectors.")

        if not isinstance(key, Prisoner) and is_bool_indexer(key):
            raise Exception("df[bool_vec] cannot be accepted because len(df) can be leaked depending on len(bool_vec).")

        if isinstance(value, Prisoner) and not isinstance(value, (DataFrame, Series)):
            raise Exception("Sensitive values (other than DataFrame and Series) cannot be assigned to dataframe.")

        if isinstance(key, Prisoner) and is_bool_indexer(key._value) and isinstance(value, Series):
            raise Exception("Sensitive Series cannot be assigned to filtered rows because Series is treated as columns here.")

        if is_2d_array(value):
            raise Exception("2D array cannot be assigned because len(df) can be leaked.")

        if not isinstance(key, Prisoner) and not is_bool_indexer(key) and \
                not isinstance(value, Prisoner) and is_list_like(value) and not isinstance(value, (_pd.DataFrame, _pd.Series)):
            raise Exception("List-like values (other than DataFrame and Series) cannot be assigned because len(df) can be leaked.")

        self._value[unwrap_prisoner(key)] = unwrap_prisoner(value)

    def __eq__(self, other):
        if isinstance(other, DataFrame):
            # FIXME: length leakage problem
            return DataFrame(data=self._value == other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return DataFrame(data=self._value == other)

    def __ne__(self, other):
        if isinstance(other, DataFrame):
            # FIXME: length leakage problem
            return DataFrame(data=self._value != other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return DataFrame(data=self._value != other)

    def __lt__(self, other):
        if isinstance(other, DataFrame):
            # FIXME: length leakage problem
            return DataFrame(data=self._value < other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return DataFrame(data=self._value < other)

    def __le__(self, other):
        if isinstance(other, DataFrame):
            # FIXME: length leakage problem
            return DataFrame(data=self._value <= other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return DataFrame(data=self._value <= other)

    def __gt__(self, other):
        if isinstance(other, DataFrame):
            # FIXME: length leakage problem
            return DataFrame(data=self._value > other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return DataFrame(data=self._value > other)

    def __ge__(self, other):
        if isinstance(other, DataFrame):
            # FIXME: length leakage problem
            return DataFrame(data=self._value >= other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against dataframe because len(df) can be leaked.")
        else:
            return DataFrame(data=self._value >= other)

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
            return DataFrame(data=self._value.reset_index(*args, **kwargs))

class Series(Prisoner):
    def __init__(self, *args, **kwargs):
        super().__init__(value=_pd.Series(*args, **kwargs), sensitivity=-1)

    def __len__(self):
        # We cannot return Prisoner() here because len() must be an integer value
        raise Exception("len(ser) is not supported. Use ser.shape[0] or ser.size instead.")

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise Exception("ser[slice] cannot be accepted because len(ser) can be leaked depending on len(slice).")

        if not isinstance(key, Prisoner):
            raise Exception("ser[bool_vec] cannot be accepted because len(ser) can be leaked depending on len(bool_vec).")

        if not is_bool_indexer(key._value):
            raise Exception("ser[key] is not allowed for sensitive keys other than boolean vectors.")

        return Series(data=self._value.__getitem__(unwrap_prisoner(key)))

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            raise Exception("ser[slice] cannot be accepted because len(ser) can be leaked depending on len(slice).")

        if not isinstance(key, Prisoner):
            raise Exception("ser[bool_vec] cannot be accepted because len(ser) can be leaked depending on len(bool_vec).")

        if not is_bool_indexer(key._value):
            raise Exception("ser[key] is not allowed for sensitive keys other than boolean vectors.")

        if isinstance(value, Prisoner) and not isinstance(value, (DataFrame, Series)):
            raise Exception("Sensitive values (other than DataFrame and Series) cannot be assigned to dataframe.")

        if isinstance(value, Series):
            raise Exception("Sensitive Series cannot be assigned to filtered rows because Series is treated as columns here.")

        if is_2d_array(value):
            raise Exception("2D array cannot be assigned because len(df) can be leaked.")

        self._value[unwrap_prisoner(key)] = unwrap_prisoner(value)

    def __eq__(self, other):
        if isinstance(other, Series):
            # FIXME: length leakage problem
            return Series(data=self._value == other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return Series(data=self._value == other)

    def __ne__(self, other):
        if isinstance(other, Series):
            # FIXME: length leakage problem
            return Series(data=self._value != other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return Series(data=self._value != other)

    def __lt__(self, other):
        if isinstance(other, Series):
            # FIXME: length leakage problem
            return Series(data=self._value < other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return Series(data=self._value < other)

    def __le__(self, other):
        if isinstance(other, Series):
            # FIXME: length leakage problem
            return Series(data=self._value <= other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return Series(data=self._value <= other)

    def __gt__(self, other):
        if isinstance(other, Series):
            # FIXME: length leakage problem
            return Series(data=self._value > other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return Series(data=self._value > other)

    def __ge__(self, other):
        if isinstance(other, Series):
            # FIXME: length leakage problem
            return Series(data=self._value >= other._value)
        elif isinstance(other, Prisoner):
            raise Exception("Sensitive values cannot be used for comparison.")
        elif is_list_like(other):
            raise Exception("List-like values cannot be compared against series because len(ser) can be leaked.")
        else:
            return Series(data=self._value >= other)

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
            return Series(data=self._value.reset_index(*args, **kwargs))
        else:
            return DataFrame(data=self._value.reset_index(*args, **kwargs))

def read_csv(*args, **kwargs):
    return DataFrame(data=_pd.read_csv(*args, **kwargs))
