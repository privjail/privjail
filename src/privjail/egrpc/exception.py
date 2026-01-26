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
import json
import builtins
import importlib

class RemoteError(Exception):
    pass

def serialize_exception(exc: Exception) -> tuple[str, str]:
    info = {
        "type": f"{type(exc).__module__}.{type(exc).__name__}",
        "message": str(exc),
    }
    return ("x-exception-info", json.dumps(info))

def deserialize_exception(metadata: list[tuple[str, str]]) -> Exception:
    metadata_dict = dict(metadata)
    info_json = metadata_dict.get("x-exception-info")
    if info_json is None:
        return RemoteError("Unknown remote error")

    info = json.loads(info_json)
    exc_cls = _get_exception_class(info["type"])
    return exc_cls(info["message"])

def _get_exception_class(type_name: str) -> type[Exception]:
    parts = type_name.rsplit(".", 1)
    if len(parts) == 2:
        module_name, class_name = parts
        try:
            module = importlib.import_module(module_name)
            cls: type[Exception] | None = getattr(module, class_name, None)
            if cls is not None and isinstance(cls, type) and issubclass(cls, Exception):
                return cls
        except (ImportError, AttributeError):
            pass

    # Fallback: try builtins (ValueError, TypeError, etc.)
    class_name = type_name.split(".")[-1]
    builtin_cls: type[Exception] | None = getattr(builtins, class_name, None)
    if builtin_cls is not None and isinstance(builtin_cls, type) and issubclass(builtin_cls, Exception):
        return builtin_cls

    return RemoteError
