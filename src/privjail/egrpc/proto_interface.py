from typing import get_type_hints, get_args, Union, List, Tuple, Dict, TypeVar, Callable, Any, ParamSpec, Type
from types import UnionType, ModuleType

from . import names
from .util import is_type_match, get_function_typed_params, get_function_return_type, get_method_typed_params, get_method_return_type, normalize_args, TypeHint, my_get_origin
from .compiler import proto_primitive_type_mapping, proto_dataclass_type_mapping, proto_remoteclass_type_mapping
from .instance_ref import InstanceRefType, get_ref_from_instance, get_instance_from_ref

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

ProtoMsg = Any

dynamic_pb2: ModuleType | None = None

def init_proto(module: ModuleType) -> None:
    global dynamic_pb2
    dynamic_pb2 = module

def get_proto_module() -> ModuleType:
    global dynamic_pb2
    assert dynamic_pb2 is not None
    return dynamic_pb2

def new_proto_function_request(func: Callable[P, R]) -> ProtoMsg:
    return getattr(get_proto_module(), names.proto_function_req_name(func))()

def new_proto_function_response(func: Callable[P, R]) -> ProtoMsg:
    return getattr(get_proto_module(), names.proto_function_res_name(func))()

def pack_proto_function_request(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> ProtoMsg:
    typed_params = get_function_typed_params(func)
    normalized_args = normalize_args(func, *args, **kwargs)

    msg = new_proto_function_request(func)

    for param_name, arg in normalized_args.items():
        set_proto_field(msg, param_name, typed_params[param_name], arg)

    return msg

def unpack_proto_function_request(func: Callable[P, R], msg: ProtoMsg) -> dict[str, Any]:
    typed_params = get_function_typed_params(func)
    return {param_name: get_proto_field(msg, param_name, type_hint) \
            for param_name, type_hint in typed_params.items()}

def pack_proto_function_response(func: Callable[P, R], obj: R) -> ProtoMsg:
    msg = new_proto_function_response(func)
    return_type = get_function_return_type(func)
    set_proto_field(msg, "return", return_type, obj)
    return msg

def unpack_proto_function_response(func: Callable[P, R], msg: ProtoMsg) -> Any:
    return_type = get_function_return_type(func)
    return get_proto_field(msg, "return", return_type)

def new_proto_method_request(cls: Type[T], method: Callable[P, R]) -> ProtoMsg:
    return getattr(get_proto_module(), names.proto_method_req_name(cls, method))()

def new_proto_method_response(cls: Type[T], method: Callable[P, R]) -> ProtoMsg:
    return getattr(get_proto_module(), names.proto_method_res_name(cls, method))()

def pack_proto_method_request(cls: Type[T], method: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> ProtoMsg:
    typed_params = get_method_typed_params(cls, method)
    normalized_args = normalize_args(method, *args, **kwargs)

    msg = new_proto_method_request(cls, method)

    for i, (param_name, arg) in enumerate(normalized_args.items()):
        if i == 0 and method.__name__ == "__init__":
            continue
        set_proto_field(msg, param_name, typed_params[param_name], arg)

    return msg

def unpack_proto_method_request(cls: Type[T], method: Callable[P, R], msg: ProtoMsg) -> dict[str, Any]:
    typed_params = get_method_typed_params(cls, method)

    if method.__name__ == "__init__":
        typed_params = dict(list(typed_params.items())[1:])

    return {param_name: get_proto_field(msg, param_name, type_hint) \
            for param_name, type_hint in typed_params.items()}

def pack_proto_method_response(cls: Type[T], method: Callable[P, R], obj: R) -> ProtoMsg:
    msg = new_proto_method_response(cls, method)

    if method.__name__ == "__init__":
        instance_ref = get_ref_from_instance(cls, obj)
        set_proto_field(msg, "return", InstanceRefType, instance_ref)
        return msg

    elif method.__name__ == "__del__":
        return msg

    else:
        return_type = get_method_return_type(cls, method)
        set_proto_field(msg, "return", return_type, obj)
        return msg

def unpack_proto_method_response(cls: Type[T], method: Callable[P, R], msg: ProtoMsg) -> Any:
    if method.__name__ == "__init__":
        return get_proto_field(msg, "return", InstanceRefType)

    elif method.__name__ == "__del__":
        return None

    else:
        return_type = get_method_return_type(cls, method)
        return get_proto_field(msg, "return", return_type)

def get_proto_field(proto_msg: ProtoMsg, param_name: str, type_hint: TypeHint) -> Any:
    type_origin = my_get_origin(type_hint)
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
            instance_ref = getattr(proto_msg, param_name).ref
            return get_instance_from_ref(cls, instance_ref)

        else:
            raise TypeError(f"Type '{type_hint}' is unknown.")

    elif type_origin in (Union, UnionType):
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

def get_proto_repeated_field(repeated_container: Any, param_name: str, type_hint: TypeHint) -> list[Any]:
    if my_get_origin(type_hint) is None:
        # RepeatedScalarContainer
        return list(repeated_container)
    else:
        # RepeatedCompositeContainer
        objs = []
        for child_proto_msg in repeated_container:
            WrapperMsg = type("WrapperMessage", (), {param_name: child_proto_msg})
            objs.append(get_proto_field(WrapperMsg, param_name, type_hint))
        return objs

def set_proto_field(proto_msg: ProtoMsg, param_name: str, type_hint: TypeHint, obj: Any) -> None:
    type_origin = my_get_origin(type_hint)
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
            cls = type_hint
            instance_ref = get_ref_from_instance(cls, obj)
            getattr(proto_msg, param_name).ref = instance_ref

        else:
            raise TypeError(f"Type '{type_hint}' is unknown.")

    elif type_origin in (Union, UnionType):
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

def set_proto_repeated_field(repeated_container: Any, param_name: str, type_hint: TypeHint, objs: list[Any]) -> None:
    if my_get_origin(type_hint) is None:
        # RepeatedScalarContainer
        repeated_container.extend(objs)
    else:
        # RepeatedCompositeContainer
        for o in objs:
            # https://stackoverflow.com/questions/57222346/how-to-get-type-contained-by-protobufs-repeatedcompositecontainer-or-repeatedsc
            child_proto_msg = repeated_container.add()
            WrapperMsg = type("WrapperMessage", (), {param_name: child_proto_msg})
            set_proto_field(WrapperMsg, param_name, type_hint, o)
