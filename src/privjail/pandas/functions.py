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
from typing import TypeVar, Any, Sequence
import json
import itertools

import pandas as _pd

from .. import egrpc
from ..util import DPError, ElementType, realnum
from ..alignment import assert_axis_signature
from ..prisoner import SensitiveInt
from ..realexpr import RealExpr
from ..accountants import BudgetType, Accountant, PureAccountant, ApproxAccountant
from .domain import CategoryDomain, normalize_column_schema, apply_column_schema, column_schema2domain
from .series import PrivSeries
from .dataframe import PrivDataFrame, SensitiveDataFrame

T = TypeVar("T")

@egrpc.function
def read_csv(filepath     : str,
             schemapath   : str | None        = None,
             *,
             accountant   : str | None        = "approx",
             budget_limit : BudgetType | None = None,
             ) -> PrivDataFrame:
    # TODO: more vaildation for the input data
    df = _pd.read_csv(filepath)

    if schemapath is not None:
        with open(schemapath, "r") as f:
            schema = json.load(f)
    else:
        schema = dict()

    domains = dict()
    user_key = None
    for col in df.columns:
        if not isinstance(col, str):
            raise ValueError("Column name must be a string.")

        if col in schema:
            col_schema = schema[col]
        else:
            col_schema = dict(type="string" if df.dtypes[col] == "object" else df.dtypes[col])

        is_user_key = col_schema["user_key"] if "user_key" in col_schema else False
        if is_user_key:
            user_key = col
            # TODO: handle duplicates of user keys

        col_schema = normalize_column_schema(col_schema)

        df[col] = apply_column_schema(df[col], col_schema, col)

        domains[col] = column_schema2domain(col_schema)

    if accountant == "pure":
        acc = PureAccountant(budget_limit=budget_limit)
    elif accountant == "approx":
        acc = ApproxAccountant(budget_limit=budget_limit)
    else:
        raise ValueError(f"Unknown accountant: '{accountant}'")

    acc.set_as_root(name=filepath)

    return PrivDataFrame(data       = df,
                         domains    = domains,
                         distance   = RealExpr(1),
                         user_key   = user_key,
                         accountant = acc)

@egrpc.function
def crosstab(index        : PrivSeries[ElementType], # TODO: support Sequence[PrivSeries[ElementType]]
             columns      : PrivSeries[ElementType], # TODO: support Sequence[PrivSeries[ElementType]]
             values       : PrivSeries[ElementType] | None = None,
             rownames     : list[str] | None               = None,
             colnames     : list[str] | None               = None,
             *,
             aggfunc      : None                           = None,
             margins      : bool                           = False,
             margins_name : str                            = "All",
             dropna       : bool                           = True,
             normalize    : bool | str | int               = False, # TODO: support Literal["all", "index", "columns", 0, 1] in egrpc
             ) -> SensitiveDataFrame:
    index._assert_not_uldp()
    columns._assert_not_uldp()

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
    else:
        raise DPError("Series for crosstab() must be of a categorical type")

    if isinstance(columns.domain, CategoryDomain):
        colvalues = columns.domain.categories
    else:
        raise DPError("Series for crosstab() must be of a categorical type")

    # if not dropna and (not any(_np.isnan(rowvalues)) or not any(_np.isnan(colvalues))):
    #     # TODO: consider handling for pd.NA
    #     warnings.warn("Counts for NaN will be dropped from the result because NaN is not included in `rowvalues`/`colvalues`", UserWarning)

    assert_axis_signature(index, columns)

    counts = _pd.crosstab(index._value, columns._value,
                          values=None, rownames=rownames, colnames=colnames,
                          aggfunc=None, margins=False, margins_name=margins_name,
                          dropna=dropna, normalize=False)

    # Select only the specified values and fill non-existent counts with 0
    counts = counts.reindex(list(rowvalues), axis="index") \
                   .reindex(list(colvalues), axis="columns") \
                   .fillna(0).astype(int)

    return SensitiveDataFrame(counts, distance_group="df", distance=index.distance, parents=[index, columns])

# TODO: change multifunction -> function by type checking in egrpc.function
@egrpc.multifunction
def cut(x              : PrivSeries[Any],
        bins           : Sequence[realnum],
        right          : bool                                = True,
        labels         : Sequence[ElementType] | bool | None = None,
        retbins        : bool                                = False,
        precision      : int                                 = 3,
        include_lowest : bool                                = False
        # TODO: add more parameters
        ) -> PrivSeries[Any]:
    x._assert_not_uldp()

    ser = _pd.cut(x._value, bins=bins, right=right, labels=labels, retbins=retbins, precision=precision, include_lowest=include_lowest) # type: ignore

    new_domain = CategoryDomain(categories=list(ser.dtype.categories))

    return PrivSeries[Any](ser, domain=new_domain, distance=x.distance, parents=[x], preserve_row=True)
