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

from typing import TYPE_CHECKING, Any

import egrpc
import jax
import jax.numpy as jnp
import numpy as np

from ..util import DPError

if TYPE_CHECKING:
    from .array import PrivArray


@egrpc.dataclass
class JaxArrayPayload:
    data: bytes
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def pack(cls, array: jax.Array) -> JaxArrayPayload:
        host = np.asarray(jax.device_get(array))
        if not host.flags.c_contiguous:
            host = np.ascontiguousarray(host)
        return cls(data=host.tobytes(), shape=host.shape, dtype=str(host.dtype))

    def unpack(self) -> jax.Array:
        host = np.frombuffer(self.data, dtype=np.dtype(self.dtype)).reshape(self.shape)
        return jnp.asarray(host)


egrpc.register_type(jax.Array, JaxArrayPayload)


def require_static(value: Any, name: str) -> None:
    if isinstance(value, jax.core.Tracer):
        raise TypeError(
            f"{name} must be static; mark its function "
            "argument with static_argnums or static_argnames."
        )


def assert_same_accountant(*arrays: PrivArray) -> None:
    if arrays and any(
        array._accountant is not arrays[0]._accountant
        for array in arrays[1:]
    ):
        raise DPError("Private operands must share one accountant.")


def assert_same_distance_and_accountant(*arrays: PrivArray) -> None:
    assert_same_accountant(*arrays)
    if arrays and any(
        not array._distance.structurally_equal(arrays[0]._distance)
        for array in arrays[1:]
    ):
        raise DPError(
            "Private operands must have identical distance expressions."
        )
