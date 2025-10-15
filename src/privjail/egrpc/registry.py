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
from typing import Any, Protocol, Type
from typing import Self

from .util import TypeHint

class PayloadType(Protocol):
    @classmethod
    def pack(cls, value: Any) -> Self:
        ...

    def unpack(self) -> Any:
        ...

_registry: dict[Type[Any], Type[PayloadType]] = {}

def register_type(python_type: Type[Any], payload_cls: Type[PayloadType]) -> None:
    global _registry
    _registry[python_type] = payload_cls

def get_handler_for_type(type_hint: TypeHint) -> Type[PayloadType] | None:
    global _registry
    for python_type, handler in _registry.items():
        if isinstance(type_hint, type) and issubclass(type_hint, python_type):
            return handler
    return None
