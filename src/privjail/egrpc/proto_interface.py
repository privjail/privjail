from typing import get_type_hints, get_origin, get_args, Union, List, Tuple, Dict
import inspect
import types
import weakref

from . import names
from .compiler import InstanceRef, proto_primitive_type_mapping, proto_dataclass_type_mapping, proto_remoteclass_type_mapping
from .util import egrpc_mode, is_type_match, get_function_typed_params, get_function_return_type, get_method_self_name, get_method_typed_params, get_method_return_type

dynamic_pb2 = None

def init_proto(module):
    global dynamic_pb2
    dynamic_pb2 = module

def get_proto_module():
    global dynamic_pb2
    assert dynamic_pb2 is not None
    return dynamic_pb2

def new_proto_function_request(func):
    return getattr(get_proto_module(), names.proto_function_req_name(func))()

def new_proto_function_response(func):
    return getattr(get_proto_module(), names.proto_function_res_name(func))()

def pack_proto_function_request(func, *args, **kwargs):
    type_hints = get_type_hints(func)

    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    msg = new_proto_function_request(func)

    for param_name, arg in bound_args.arguments.items():
        set_proto_field(msg, param_name, type_hints[param_name], arg)

    return msg

def unpack_proto_function_request(func, msg):
    typed_params = get_function_typed_params(func)
    return {param_name: get_proto_field(msg, param_name, type_hint) \
            for param_name, type_hint in typed_params.items()}

def pack_proto_function_response(func, obj):
    msg = new_proto_function_response(func)
    return_type = get_function_return_type(func)
    set_proto_field(msg, "return", return_type, obj)
    return msg

def unpack_proto_function_response(func, msg):
    return_type = get_function_return_type(func)
    return get_proto_field(msg, "return", return_type)

def new_proto_method_request(cls, method):
    return getattr(get_proto_module(), names.proto_method_req_name(cls, method))()

def new_proto_method_response(cls, method):
    return getattr(get_proto_module(), names.proto_method_res_name(cls, method))()

def pack_proto_method_request(cls, method, *args, **kwargs):
    type_hints = get_type_hints(method)

    sig = inspect.signature(method)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    msg = new_proto_method_request(cls, method)

    for i, (param_name, arg) in enumerate(bound_args.arguments.items()):
        if i == 0:
            if method.__name__ != "__init__":
                set_proto_field(msg, param_name, InstanceRef, arg.__instance_ref)
        else:
            set_proto_field(msg, param_name, type_hints[param_name], arg)

    return msg

def unpack_proto_method_request(cls, method, msg):
    typed_params = get_method_typed_params(cls, method)
    args = {param_name: get_proto_field(msg, param_name, type_hint) \
            for param_name, type_hint in typed_params.items()}

    if method.__name__ == "__init__":
        return args

    else:
        self_name = get_method_self_name(cls, method)
        instance_ref = get_proto_field(msg, self_name, InstanceRef)
        obj = cls.__instance_map[instance_ref]
        return {self_name: obj, **args}

def pack_proto_method_response(cls, method, obj):
    msg = new_proto_method_response(cls, method)

    if method.__name__ == "__init__":
        set_proto_field(msg, "return", cls, obj)
        return msg

    elif method.__name__ == "__del__":
        return msg

    else:
        return_type = get_method_return_type(cls, method)
        set_proto_field(msg, "return", return_type, obj)
        return msg

def unpack_proto_method_response(cls, method, msg):
    if method.__name__ == "__init__":
        return get_proto_field(msg, "return", InstanceRef)

    elif method.__name__ == "__del__":
        return None

    else:
        return_type = get_method_return_type(cls, method)
        return get_proto_field(msg, "return", return_type)

def assign_instance_ref(cls, obj):
    instance_ref = cls.__instance_count
    cls.__instance_count += 1

    obj.__instance_ref = instance_ref
    cls.__instance_map[instance_ref] = obj

def get_proto_field(proto_msg, param_name, type_hint):
    type_origin = get_origin(type_hint)
    type_args = get_args(type_hint)

    if type_origin is None:
        if type_hint is type(None):
            return None

        elif type_hint in proto_primitive_type_mapping:
            return getattr(proto_msg, param_name)

        elif type_hint in proto_dataclass_type_mapping:
            proto_class_msg = getattr(proto_msg, param_name)
            args = {member_name: get_proto_field(proto_class_msg, member_name, th)
                    for member_name, th in get_type_hints(type_hint).items()}
            return type_hint(**args)

        elif type_hint in proto_remoteclass_type_mapping:
            cls = type_hint
            instance_ref = getattr(proto_msg, param_name)

            if egrpc_mode == "server":
                assert instance_ref in cls.__instance_map
                return cls.__instance_map[instance_ref]

            elif egrpc_mode == "client":
                if instance_ref in cls.__instance_map:
                    return cls.__instance_map[instance_ref]()
                else:
                    obj = object.__new__(cls)
                    obj.__instance_ref = instance_ref
                    cls.__instance_map[instance_ref] = weakref.ref(obj)
                    return obj

            else:
                raise Exception

        else:
            raise TypeError(f"Type '{type_hint}' is unknown.")

    elif type_origin in (Union, types.UnionType):
        child_proto_msg = getattr(proto_msg, param_name)
        for i, th in enumerate(type_args):
            if child_proto_msg.HasField(f"member{i}"):
                return get_proto_field(child_proto_msg, f"member{i}", th)
        raise ValueError(f"Parameter '{param_name}' is empty.")

    elif type_origin in (tuple, Tuple):
        child_proto_msg = getattr(proto_msg, param_name)
        objs = []
        for i, th in enumerate(type_args):
            objs.append(get_proto_field(child_proto_msg, f"item{i}", th))
        return tuple(objs)

    elif type_origin in (list, List):
        repeated_container = getattr(proto_msg, param_name).elements
        return get_proto_repeated_field(repeated_container, param_name, type_args[0])

    elif type_origin in (dict, Dict):
        repeated_container_k = getattr(proto_msg, param_name).keys
        repeated_container_v = getattr(proto_msg, param_name).values
        keys = get_proto_repeated_field(repeated_container_k, param_name, type_args[0])
        values = get_proto_repeated_field(repeated_container_v, param_name, type_args[1])
        assert len(keys) == len(values)
        return dict(zip(keys, values))

    else:
        raise Exception

def set_proto_repeated_field(repeated_container, param_name, type_hint, objs):
    if get_origin(type_hint) is None:
        # RepeatedScalarContainer
        repeated_container.extend(objs)
    else:
        # RepeatedCompositeContainer
        for o in objs:
            # https://stackoverflow.com/questions/57222346/how-to-get-type-contained-by-protobufs-repeatedcompositecontainer-or-repeatedsc
            child_proto_msg = repeated_container.add()
            WrapperMsg = type("WrapperMessage", (), {param_name: child_proto_msg})
            set_proto_field(WrapperMsg, param_name, type_hint, o)

def set_proto_field(proto_msg, param_name, type_hint, obj):
    type_origin = get_origin(type_hint)
    type_args = get_args(type_hint)

    if type_origin is None:
        if type_hint is type(None):
            setattr(proto_msg, param_name, True)

        elif type_hint in proto_primitive_type_mapping:
            setattr(proto_msg, param_name, obj)

        elif type_hint in proto_dataclass_type_mapping:
            proto_class_msg = getattr(proto_msg, param_name)
            for member_name, th in get_type_hints(type_hint).items():
                set_proto_field(proto_class_msg, member_name, th, getattr(obj, member_name))

        elif type_hint in proto_remoteclass_type_mapping:
            if egrpc_mode == "server" and not hasattr(obj, "__instance_ref"):
                assign_instance_ref(type_hint, obj)
            setattr(proto_msg, param_name, obj.__instance_ref)

        else:
            raise TypeError(f"Type '{type_hint}' is unknown.")

    elif type_origin in (Union, types.UnionType):
        child_proto_msg = getattr(proto_msg, param_name)
        for i, th in enumerate(type_args):
            if is_type_match(obj, th):
                set_proto_field(child_proto_msg, f"member{i}", th, obj)
                return
        raise TypeError(f"Type '{type(obj)}' of parameter '{param_name}' does not match any of {type_hint}.")

    elif type_origin in (tuple, Tuple):
        child_proto_msg = getattr(proto_msg, param_name)
        for i, (th, o) in enumerate(zip(type_args, obj)):
            set_proto_field(child_proto_msg, f"item{i}", th, o)

    elif type_origin in (list, List):
        repeated_container = getattr(proto_msg, param_name).elements
        set_proto_repeated_field(repeated_container, param_name, type_args[0], obj)

    elif type_origin in (dict, Dict):
        repeated_container_k = getattr(proto_msg, param_name).keys
        repeated_container_v = getattr(proto_msg, param_name).values
        set_proto_repeated_field(repeated_container_k, param_name, type_args[0], obj.keys())
        set_proto_repeated_field(repeated_container_v, param_name, type_args[1], obj.values())

    else:
        raise Exception

def get_proto_repeated_field(repeated_container, param_name, type_hint):
    if get_origin(type_hint) is None:
        # RepeatedScalarContainer
        return list(repeated_container)
    else:
        # RepeatedCompositeContainer
        objs = []
        for child_proto_msg in repeated_container:
            WrapperMsg = type("WrapperMessage", (), {param_name: child_proto_msg})
            objs.append(get_proto_field(WrapperMsg, param_name, type_hint))
        return objs
