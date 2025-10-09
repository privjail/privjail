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

import numpy as _np

from .. import egrpc
from ..prisoner import SensitiveFloat
from ..util import DPError
from .array import PrivNDArray, SensitiveNDArray

# TODO: support `axis` parameter
@egrpc.multifunction
def norm(x: PrivNDArray, ord: int | None = None) -> SensitiveFloat:
    ord_value = 2 if ord is None else ord
    if ord_value not in (1, 2):
        raise NotImplementedError("Only ord=1 or ord=2 (or None) are supported.")

    domain = x.domain
    if ord_value == 1 and domain.norm_type != "l1":
        raise DPError("Domain norm type is not L1. Call clip_norm with ord=1 first.")
    if ord_value == 2 and domain.norm_type not in ("l2", "l1"):
        raise DPError("Domain norm type is neither L2 nor L1; cannot compute L2 norm.")
    bound = domain.norm_bound
    if bound is None:
        raise DPError("Norm bound is not set. Use clip_norm() before calling norm().")
    if x.distance.is_inf():
        raise DPError("Unbounded distance")

    value = float(_np.linalg.norm(x._value, ord=ord_value))
    new_distance = x.distance * float(bound)
    return SensitiveFloat(value, distance=new_distance, parents=[x])

@norm.register
def _(x: SensitiveNDArray, ord: int | None = None) -> SensitiveFloat:
    ord_value = 2 if ord is None else ord
    if ord_value not in (1, 2):
        raise NotImplementedError("Only ord=1 or ord=2 (or None) are supported.")
    if ord_value == 1 and x._norm_type != "l1":
        raise DPError("Requested `ord` does not match L1 sensitivity.")
    if ord_value == 2 and x._norm_type not in ("l2", "l1"):
        raise DPError("Requested `ord` does not match available sensitivity.")

    value = float(_np.linalg.norm(x._value, ord=ord_value))

    new_distance = x.distance

    return SensitiveFloat(value, distance=new_distance, parents=[x])
