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

from .. import egrpc
from ..util import realnum
from ..distance import Distance
from ..prisoner import SensitiveInt, SensitiveFloat

ElementType = realnum | str | bool

PTag = int

ptag_count = 0

def new_ptag() -> PTag:
    global ptag_count
    ptag_count += 1
    return ptag_count

@egrpc.function
def total_max_distance(prisoners: list[SensitiveInt | SensitiveFloat]) -> realnum:
    return sum([x.distance for x in prisoners], start=Distance(0)).max()
