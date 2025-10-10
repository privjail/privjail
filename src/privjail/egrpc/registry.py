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
from typing import Any, Callable, NamedTuple, Type

from .util import TypeHint

class CustomTypeHandler(NamedTuple):
    python_type    : Type[Any]
    surrogate_type : Type[Any]
    to_surrogate   : Callable[[Any], Any]
    from_surrogate : Callable[[Any], Any]

_registry: dict[Type[Any], CustomTypeHandler] = {}

def register_type(python_type    : Type[Any],
                  surrogate_type : Type[Any],
                  to_surrogate   : Callable[[Any], Any],
                  from_surrogate : Callable[[Any], Any]) -> None:
    global _registry
    _registry[python_type] = CustomTypeHandler(python_type, surrogate_type, to_surrogate, from_surrogate)

def get_handler_for_type(type_hint: TypeHint) -> CustomTypeHandler | None:
    global _registry
    for python_type, handler in _registry.items():
        if isinstance(type_hint, type) and issubclass(type_hint, python_type):
            return handler
    return None
