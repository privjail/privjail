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
        instance_ref = getattr(cls, "__instance_count")
        setattr(cls, "__instance_count", instance_ref + 1)
        assign_ref_to_instance(cls, obj, instance_ref, on_server)

    return _get_instance_ref(obj)

def get_instance_from_ref(cls: Type[T], instance_ref: InstanceRefType, type_hint: TypeHint, on_server: bool) -> T:
    if on_server:
        instance_map_server = _get_instance_map_server(cls)
        assert instance_ref in instance_map_server
        return instance_map_server[instance_ref]

    else:
        instance_map_client = _get_instance_map_client(cls)

        if instance_ref in instance_map_client:
            obj = instance_map_client[instance_ref]()
            assert obj is not None
            return obj

        else:
            obj = object.__new__(cls)

            # If this type is a generic type, set `__orig_class__` attribute
            # so that this information is used for type-based dispatch (by multimethod.multidispatch)
            if len(get_args(type_hint)) > 0:
                setattr(obj, "__orig_class__", type_hint)

            assign_ref_to_instance(cls, obj, instance_ref, on_server)
            return obj

def assign_ref_to_instance(cls: Type[T], obj: T, instance_ref: InstanceRefType, on_server: bool) -> None:
    setattr(obj, "__instance_ref", instance_ref)

    if on_server:
        instance_map_server = _get_instance_map_server(cls)
        assert instance_ref not in instance_map_server
        instance_map_server[instance_ref] = obj

    else:
        # use weakref so that obj is garbage collected regardless of instance map
        instance_map_client = _get_instance_map_client(cls)
        assert instance_ref not in instance_map_client
        instance_map_client[instance_ref] = weakref.ref(obj)

def del_instance(cls: Type[T], obj: T) -> None:
    instance_map = _get_instance_map_server(cls)
    instance_ref = _get_instance_ref(obj)
    del instance_map[instance_ref]
