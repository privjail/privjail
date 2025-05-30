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
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import field

import pandas as _pd

from .. import egrpc
from ..util import realnum
from .util import ElementType

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

    def type(self) -> type:
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
