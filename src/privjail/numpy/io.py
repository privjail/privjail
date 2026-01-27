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
import json

import numpy as _np
import numpy.typing as _npt

from .. import egrpc
from ..realexpr import RealExpr
from ..accountants import BudgetType, Accountant, PureDPAccountant, ApproxDPAccountant
from .array import PrivNDArray
from .domain import NDArrayDomain

def _find_distance_axis(shape: list[int | None] | None) -> int:
    if shape is None:
        return 0
    null_indices = [i for i, dim in enumerate(shape) if dim is None]
    if len(null_indices) != 1:
        raise ValueError("shape must contain exactly one null for distance_axis")
    return null_indices[0]

@egrpc.function
def load(file         : str,
         schema_path  : str | None        = None,
         *,
         accountant   : str | None        = "approx",
         budget_limit : BudgetType | None = None,
         ) -> dict[str, PrivNDArray | _npt.NDArray[Any]]:
    npz_file = _np.load(file)

    schema: dict[str, Any] = {}
    if schema_path is not None:
        with open(schema_path, "r") as f:
            schema = json.load(f)

    acc: Accountant[Any]
    if accountant == "pure":
        acc = PureDPAccountant(budget_limit=PureDPAccountant.normalize_budget(budget_limit))
    elif accountant == "approx":
        acc = ApproxDPAccountant(budget_limit=ApproxDPAccountant.normalize_budget(budget_limit))
    else:
        raise ValueError(f"Unknown accountant: '{accountant}'")

    acc.set_as_root(name=file)

    result: dict[str, PrivNDArray | _npt.NDArray[Any]] = {}
    signature_map: dict[str, int] = {}

    for name in npz_file.files:
        arr = npz_file[name]
        arr_schema = schema.get(name, {})

        if arr_schema.get("public", False):
            result[name] = arr
            continue

        schema_dtype = arr_schema.get("dtype")
        if schema_dtype is not None and _np.dtype(schema_dtype) != arr.dtype:
            raise ValueError(f"dtype mismatch for '{name}': schema={schema_dtype}, actual={arr.dtype}")

        shape = arr_schema.get("shape")
        if shape is not None:
            if len(shape) != arr.ndim:
                raise ValueError(f"ndim mismatch for '{name}': schema={len(shape)}, actual={arr.ndim}")
            for i, (schema_dim, actual_dim) in enumerate(zip(shape, arr.shape)):
                if schema_dim is not None and schema_dim != actual_dim:
                    raise ValueError(f"shape mismatch for '{name}' at axis {i}: schema={schema_dim}, actual={actual_dim}")

        distance_axis = _find_distance_axis(shape)

        value_range_spec = arr_schema.get("value_range")
        if value_range_spec is not None:
            domain = NDArrayDomain(value_range=(float(value_range_spec[0]), float(value_range_spec[1])))
        else:
            domain = NDArrayDomain()

        priv_arr = PrivNDArray(
            value         = arr,
            distance      = RealExpr(1),
            distance_axis = distance_axis,
            domain        = domain,
            accountant    = acc,
        )

        align_sig = arr_schema.get("alignment_signature")
        if align_sig is not None:
            if align_sig not in signature_map:
                signature_map[align_sig] = priv_arr._axis_signature
            else:
                priv_arr._axis_signature = signature_map[align_sig]

        result[name] = priv_arr

    return result
