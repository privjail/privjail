# Copyright 2026 Shumpei Shiina.
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

from dataclasses import dataclass

from ..realexpr import RealExpr


ValueRange = tuple[float | None, float | None] | None


@dataclass(frozen=True)
class ArrayDomain:
    norm_type: str = "l1"
    norm_bound: RealExpr | None = None
    value_range: ValueRange = None


__all__ = ["ArrayDomain"]
