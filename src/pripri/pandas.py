from .prisoner import Prisoner
import pandas as _pd
from pandas.api.types import is_list_like
# FIXME: pandas.core.common API is not public
from pandas.core.common import is_bool_indexer

class DataFrame(Prisoner):
    def __init__(self, *args, **kwargs):
        super().__init__(value=_pd.DataFrame(*args, **kwargs), sensitivity=-1)

    def __len__(self):
        # We cannot return Prisoner() here because len() must be an integer value
        raise Exception("len(df) is not supported. Use df.shape[0] instead.")

    def __getitem__(self, key):
        if isinstance(key, Prisoner):
            if is_list_like(key._value):
                # df[bool_vec] : select rows by boolean vector
                # FIXME: length leakage problem
                #        Strictly, we need to check if the data source of `key` is this dataframe.
                #        Otherwise, len(df) can be leaked by providing `key` of a certain length.
                return DataFrame(data=self._value.__getitem__(key._value))
            else:
                raise Exception("df[key] is not allowed for sensitive keys other than boolean vectors.")

        if isinstance(key, slice):
            raise Exception("df[slice] cannot be accepted because len(df) can be leaked depending on len(slice).")

        if is_bool_indexer(key):
            raise Exception("df[bool_vec] cannot be accepted because len(df) can be leaked depending on len(bool_vec).")

        data = self._value.__getitem__(key)
        if isinstance(data, _pd.DataFrame):
            # df[[col1, col2, ...]] : select multiple columns
            return DataFrame(data)
        elif isinstance(data, _pd.Series):
            # df[col] : select a column
            return Series(data)
        else:
            raise RuntimeError

    def __setitem__(self, key, value):
        if isinstance(value, Series):
            # FIXME: length leakage problem
            self._value[key] = value._value
        elif isinstance(value, Prisoner):
            raise Exception("Sensitive values (other than Series) cannot be assigned to dataframe.")
        elif is_list_like(value):
            raise Exception("List-like values cannot be assigned to dataframe because len(df) can be leaked.")
        else:
            self._value[key] = value

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

class Series(Prisoner):
    def __init__(self, *args, **kwargs):
        super().__init__(value=_pd.Series(*args, **kwargs), sensitivity=-1)

    def __len__(self):
        # We cannot return Prisoner() here because len() must be an integer value
        raise Exception("len(ser) is not supported. Use ser.shape[0] or ser.size instead.")

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

def read_csv(*args, **kwargs):
    return DataFrame(data=_pd.read_csv(*args, **kwargs))
