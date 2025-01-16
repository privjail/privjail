from typing import TypeVar, Type, Any
import weakref

from .util import egrpc_mode

T = TypeVar("T")

WeakRef = weakref.ref

InstanceRefType = int

def init_remoteclass(cls: Type[T]) -> None:
    if not hasattr(cls, "__del__"):
        def __del__(self: Any) -> None:
            pass
        setattr(cls, "__del__", __del__)

    if egrpc_mode == "server":
        setattr(cls, "__instance_count", 0)
        setattr(cls, "__instance_map_server", {})
    elif egrpc_mode == "client":
        setattr(cls, "__instance_map_client", {})

def _get_instance_map_server(cls: Type[T]) -> dict[InstanceRefType, T]:
    assert egrpc_mode == "server"
    instance_map: dict[InstanceRefType, T] = getattr(cls, "__instance_map_server")
    return instance_map

def _get_instance_map_client(cls: Type[T]) -> dict[InstanceRefType, WeakRef[T]]:
    assert egrpc_mode == "client"
    instance_map: dict[InstanceRefType, WeakRef[T]] = getattr(cls, "__instance_map_client")
    return instance_map

def _get_instance_ref(obj: T) -> int:
    instance_ref: int = getattr(obj, "__instance_ref")
    return instance_ref

def get_ref_from_instance(cls: Type[T], obj: T) -> InstanceRefType:
    if egrpc_mode == "server" and not hasattr(obj, "__instance_ref"):
        instance_ref = getattr(cls, "__instance_count")
        setattr(cls, "__instance_count", instance_ref + 1)
        assign_ref_to_instance(cls, obj, instance_ref)

    return _get_instance_ref(obj)

def get_instance_from_ref(cls: Type[T], instance_ref: InstanceRefType) -> T:
    if egrpc_mode == "server":
        instance_map_server = _get_instance_map_server(cls)
        assert instance_ref in instance_map_server
        return instance_map_server[instance_ref]

    elif egrpc_mode == "client":
        instance_map_client = _get_instance_map_client(cls)
        if instance_ref in instance_map_client:
            obj = instance_map_client[instance_ref]()
            assert obj is not None
            return obj
        else:
            obj = object.__new__(cls)
            assign_ref_to_instance(cls, obj, instance_ref)
            return obj

    else:
        raise Exception

def assign_ref_to_instance(cls: Type[T], obj: T, instance_ref: InstanceRefType) -> None:
    setattr(obj, "__instance_ref", instance_ref)

    if egrpc_mode == "server":
        instance_map_server = _get_instance_map_server(cls)
        assert instance_ref not in instance_map_server
        instance_map_server[instance_ref] = obj

    elif egrpc_mode == "client":
        # use weakref so that obj is garbage collected regardless of instance map
        instance_map_client = _get_instance_map_client(cls)
        assert instance_ref not in instance_map_client
        instance_map_client[instance_ref] = weakref.ref(obj)

    else:
        raise Exception

def del_instance(cls: Type[T], obj: T) -> None:
    instance_map = _get_instance_map_server(cls)
    instance_ref = _get_instance_ref(obj)
    del instance_map[instance_ref]
