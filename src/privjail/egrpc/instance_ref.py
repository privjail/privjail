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

from typing import TypeVar, Type, Any, get_args
import weakref

from .util import TypeHint

T = TypeVar("T")

WeakRef = weakref.ref

InstanceRefType = int

def init_remoteclass(cls: Type[T]) -> None:
    if not hasattr(cls, "__del__"):
        def __del__(self: Any) -> None:
            pass
        setattr(cls, "__del__", __del__)

    setattr(cls, "__instance_count", 0)
    setattr(cls, "__instance_map_server", {})
    setattr(cls, "__instance_map_client", {})

def _get_instance_map_server(cls: Type[T]) -> dict[InstanceRefType, T]:
    instance_map: dict[InstanceRefType, T] = getattr(cls, "__instance_map_server")
    return instance_map

def _get_instance_map_client(cls: Type[T]) -> dict[InstanceRefType, WeakRef[T]]:
    instance_map: dict[InstanceRefType, WeakRef[T]] = getattr(cls, "__instance_map_client")
    return instance_map

def _get_instance_ref(obj: T) -> int:
    instance_ref: int = getattr(obj, "__instance_ref")
    return instance_ref

def get_ref_from_instance(cls: Type[T], obj: T, on_server: bool) -> InstanceRefType:
    if on_server and not hasattr(obj, "__instance_ref"):
        instance_ref: int = getattr(cls, "__instance_count")
        setattr(cls, "__instance_count", instance_ref + 1)
        setattr(obj, "__instance_ref", instance_ref)
    else:
        instance_ref = _get_instance_ref(obj)

    if on_server:
        instance_map_server = _get_instance_map_server(cls)
        instance_map_server[instance_ref] = obj

    return instance_ref

def get_instance_from_ref(cls: Type[T], instance_ref: InstanceRefType, type_hint: TypeHint, on_server: bool) -> T:
    if on_server:
        instance_map_server = _get_instance_map_server(cls)
        if instance_ref not in instance_map_server:
            raise ValueError(f"Unknown instance ref {instance_ref} for {cls.__name__}")
        return instance_map_server[instance_ref]

    else:
        instance_map_client = _get_instance_map_client(cls)

        if instance_ref in instance_map_client:
            obj = instance_map_client[instance_ref]()
            if obj is not None:
                return obj
            # Weakref target was GC'd; re-create the proxy instance.
            del instance_map_client[instance_ref]

        obj = object.__new__(cls)

        # If this type is a generic type, set `__orig_class__` attribute
        # so that this information is used for type-based dispatch (by multimethod.multidispatch)
        if len(get_args(type_hint)) > 0:
            setattr(obj, "__orig_class__", type_hint)

        assign_ref_to_instance(cls, obj, instance_ref)
        return obj

def assign_ref_to_instance(cls: Type[T], obj: T, instance_ref: InstanceRefType) -> None:
    setattr(obj, "__instance_ref", instance_ref)
    # use weakref so that obj is garbage collected regardless of instance map
    instance_map_client = _get_instance_map_client(cls)
    assert instance_ref not in instance_map_client
    instance_map_client[instance_ref] = weakref.ref(obj)

def del_instance(cls: Type[T], obj: T) -> None:
    instance_map = _get_instance_map_server(cls)
    instance_ref = _get_instance_ref(obj)
    if instance_ref in instance_map:
        del instance_map[instance_ref]
