import weakref

from .util import egrpc_mode

InstanceRefType = int

def init_remoteclass(cls):
    if not hasattr(cls, "__del__"):
        def __del__(self):
            pass
        cls.__del__ = __del__

    cls.__instance_count = 0
    cls.__instance_map = {}

def get_ref_from_instance(cls, obj):
    if egrpc_mode == "server" and not hasattr(obj, "__instance_ref"):
        instance_ref = cls.__instance_count
        cls.__instance_count += 1
        assign_ref_to_instance(cls, obj, instance_ref)

    return obj.__instance_ref

def get_instance_from_ref(cls, instance_ref):
    if egrpc_mode == "server":
        assert instance_ref in cls.__instance_map
        return cls.__instance_map[instance_ref]

    elif egrpc_mode == "client":
        if instance_ref in cls.__instance_map:
            return cls.__instance_map[instance_ref]()
        else:
            obj = object.__new__(cls)
            assign_ref_to_instance(cls, obj, instance_ref)
            return obj

    else:
        raise Exception

def assign_ref_to_instance(cls, obj, instance_ref):
    obj.__instance_ref = instance_ref

    assert instance_ref not in cls.__instance_map

    if egrpc_mode == "server":
        cls.__instance_map[instance_ref] = obj
    elif egrpc_mode == "client":
        # use weakref so that obj is garbage collected regardless of instance map
        cls.__instance_map[instance_ref] = weakref.ref(obj)
    else:
        raise Exception

def del_instance(cls, obj):
    del cls.__instance_map[obj.__instance_ref]
