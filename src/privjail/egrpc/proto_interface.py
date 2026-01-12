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

from typing import get_type_hints, get_args, Union, List, Tuple, Dict, TypeVar, Callable, Any, ParamSpec, Type
from types import EllipsisType, UnionType, ModuleType
from collections.abc import Sequence, Mapping

from . import names
from .util import is_type_match, get_function_typed_params, get_function_return_type, get_method_typed_params, get_method_return_type, normalize_args, TypeHint, my_get_origin
from .compiler import proto_primitive_type_mapping, proto_dataclass_type_mapping, proto_remoteclass_type_mapping, proto_remoteclass_id_mapping, proto_remoteclass_id_reverse_mapping, is_subclass, subclasses
from .instance_ref import InstanceRefType, get_ref_from_instance, get_instance_from_ref
from .registry import get_handler_for_type

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
        set_proto_field(msg, param_name, typed_params[param_name], arg, allow_subclass=True, on_server=False)

    return msg

def unpack_proto_function_request(func: Callable[P, R], msg: ProtoMsg) -> dict[str, Any]:
    typed_params = get_function_typed_params(func)
    return {param_name: get_proto_field(msg, param_name, type_hint, allow_subclass=True, on_server=True) \
            for param_name, type_hint in typed_params.items()}

def pack_proto_function_response(func: Callable[P, R], obj: R) -> ProtoMsg:
    msg = new_proto_function_response(func)
    return_type = get_function_return_type(func)
    set_proto_field(msg, "return", return_type, obj, allow_subclass=True, on_server=True)
    return msg

def unpack_proto_function_response(func: Callable[P, R], msg: ProtoMsg) -> Any:
    return_type = get_function_return_type(func)
    return get_proto_field(msg, "return", return_type, allow_subclass=True, on_server=False)

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
        set_proto_field(msg, param_name, typed_params[param_name], arg, allow_subclass=True, on_server=False)

    return msg

def unpack_proto_method_request(cls: Type[T], method: Callable[P, R], msg: ProtoMsg) -> dict[str, Any]:
    typed_params = get_method_typed_params(cls, method)

    if method.__name__ == "__init__":
        typed_params = dict(list(typed_params.items())[1:])

    return {param_name: get_proto_field(msg, param_name, type_hint, allow_subclass=True, on_server=True) \
            for param_name, type_hint in typed_params.items()}

def pack_proto_method_response(cls: Type[T], method: Callable[P, R], obj: R) -> ProtoMsg:
    msg = new_proto_method_response(cls, method)

    if method.__name__ == "__init__":
        instance_ref = get_ref_from_instance(cls, obj, True)
        set_proto_field(msg, "return", InstanceRefType, instance_ref, allow_subclass=True, on_server=True)
        return msg

    elif method.__name__ == "__del__":
        return msg

    else:
        return_type = get_method_return_type(cls, method)
        set_proto_field(msg, "return", return_type, obj, allow_subclass=True, on_server=True)
        return msg

def unpack_proto_method_response(cls: Type[T], method: Callable[P, R], msg: ProtoMsg) -> Any:
    if method.__name__ == "__init__":
        return get_proto_field(msg, "return", InstanceRefType, allow_subclass=True, on_server=False)

    elif method.__name__ == "__del__":
        return None

    else:
        return_type = get_method_return_type(cls, method)
        o = get_proto_field(msg, "return", return_type, allow_subclass=True, on_server=False)
        return o

def get_proto_field(proto_msg: ProtoMsg, param_name: str, type_hint: TypeHint, allow_subclass: bool, on_server: bool) -> Any:
    type_origin = my_get_origin(type_hint)
    type_args = get_args(type_hint)

    type_origin = type_hint if type_origin is None else type_origin

    if type_origin is type(None):
        return None

    elif type_origin is EllipsisType:
        return ...

    elif type_origin in proto_primitive_type_mapping:
        return getattr(proto_msg, param_name)

    elif type_origin is slice:
        msg = getattr(proto_msg, param_name)
        start = msg.start if msg.HasField("start") else None
        stop = msg.stop if msg.HasField("stop") else None
        step = msg.step if msg.HasField("step") else None
        return slice(start, stop, step)

    elif is_subclass(type_origin, proto_dataclass_type_mapping):
        proto_class_msg = getattr(proto_msg, param_name)
        proto_types = subclasses(type_origin, proto_dataclass_type_mapping)
        if not allow_subclass or len(proto_types) == 0:
            args = {member_name: get_proto_field(proto_class_msg, member_name, th, allow_subclass=True, on_server=on_server)
                    for member_name, th in get_type_hints(type_origin).items()}
            return type_origin(**args)
        else:
            for i, th in enumerate(proto_types.keys()):
                if proto_class_msg.HasField(f"class{i}"):
                    return get_proto_field(proto_class_msg, f"class{i}", th, allow_subclass=False, on_server=on_server)
            raise ValueError(f"Parameter '{param_name}' is empty.")

    elif type_origin in proto_remoteclass_type_mapping:
        cls = type_origin
        msg = getattr(proto_msg, param_name)
        instance_ref = msg.ref

        actual_cls = cls
        if allow_subclass:
            candidate_cls = proto_remoteclass_id_reverse_mapping.get(msg.type_id)
            if candidate_cls is not None and issubclass(candidate_cls, cls):
                actual_cls = candidate_cls

        return get_instance_from_ref(actual_cls, instance_ref, type_hint, on_server)

    elif type_origin in (Union, UnionType):
        child_proto_msg = getattr(proto_msg, param_name)
        for i, th in enumerate(type_args):
            if child_proto_msg.HasField(f"member{i}"):
                return get_proto_field(child_proto_msg, f"member{i}", th, allow_subclass=allow_subclass, on_server=on_server)
        raise ValueError(f"Parameter '{param_name}' is empty.")

    elif type_origin in (tuple, Tuple):
        if len(type_args) == 2 and type_args[1] is Ellipsis:
            repeated_container = getattr(proto_msg, param_name).items
            return tuple(get_proto_repeated_field(repeated_container, param_name, type_args[0], allow_subclass=allow_subclass, on_server=on_server))
        else:
            child_proto_msg = getattr(proto_msg, param_name)
            objs = []
            for i, th in enumerate(type_args):
                objs.append(get_proto_field(child_proto_msg, f"item{i}", th, allow_subclass=allow_subclass, on_server=on_server))
            return tuple(objs)

    elif type_origin in (list, List, Sequence):
        repeated_container = getattr(proto_msg, param_name).elements
        return get_proto_repeated_field(repeated_container, param_name, type_args[0], allow_subclass=type_origin is Sequence, on_server=on_server)

    elif type_origin in (dict, Dict, Mapping):
        repeated_container_k = getattr(proto_msg, param_name).keys
        repeated_container_v = getattr(proto_msg, param_name).values
        # Mapping key type is not covariant
        keys = get_proto_repeated_field(repeated_container_k, param_name, type_args[0], allow_subclass=False, on_server=on_server)
        values = get_proto_repeated_field(repeated_container_v, param_name, type_args[1], allow_subclass=type_origin is Mapping, on_server=on_server)
        assert len(keys) == len(values)
        return dict(zip(keys, values))

    else:
        handler = get_handler_for_type(type_origin)
        if handler is not None:
            payload = get_proto_field(proto_msg, param_name, handler, allow_subclass=allow_subclass, on_server=on_server)
            return payload.unpack()
        else:
            raise TypeError(f"Type {type_origin} is not supported.")

def get_proto_repeated_field(repeated_container: Any, param_name: str, type_hint: TypeHint, allow_subclass: bool, on_server: bool) -> list[Any]:
    if type_hint in proto_primitive_type_mapping:
        # RepeatedScalarContainer
        return list(repeated_container)
    else:
        # RepeatedCompositeContainer
        objs = []
        for child_proto_msg in repeated_container:
            WrapperMsg = type("WrapperMessage", (), {param_name: child_proto_msg})
            objs.append(get_proto_field(WrapperMsg, param_name, type_hint, allow_subclass=allow_subclass, on_server=on_server))
        return objs

def set_proto_field(proto_msg: ProtoMsg, param_name: str, type_hint: TypeHint, obj: Any, allow_subclass: bool, on_server: bool) -> None:
    type_origin = my_get_origin(type_hint)
    type_args = get_args(type_hint)

    type_origin = type_hint if type_origin is None else type_origin

    if type_origin is type(None):
        setattr(proto_msg, param_name, True)

    elif type_origin is EllipsisType:
        setattr(proto_msg, param_name, True)

    elif type_origin in proto_primitive_type_mapping:
        setattr(proto_msg, param_name, obj)

    elif type_origin is slice:
        msg = getattr(proto_msg, param_name)
        msg.is_present = True
        if obj.start is not None:
            msg.start = obj.start
        if obj.stop is not None:
            msg.stop = obj.stop
        if obj.step is not None:
            msg.step = obj.step

    elif is_subclass(type_origin, proto_dataclass_type_mapping):
        proto_class_msg = getattr(proto_msg, param_name)
        proto_types = subclasses(type_origin, proto_dataclass_type_mapping)
        if not allow_subclass or len(proto_types) == 0:
            for member_name, th in get_type_hints(type_origin).items():
                set_proto_field(proto_class_msg, member_name, th, getattr(obj, member_name), allow_subclass=True, on_server=on_server)
        else:
            for i, th in enumerate(proto_types.keys()):
                if type(obj) is th:
                    set_proto_field(proto_class_msg, f"class{i}", th, obj, allow_subclass=False, on_server=on_server)
                    return
            raise TypeError(f"Type '{type(obj)}' of parameter '{param_name}' does not match any of {type_hint}.")

    elif type_origin in proto_remoteclass_type_mapping:
        cls = type_origin
        actual_cls = type(obj) if allow_subclass and isinstance(obj, cls) else cls

        if actual_cls not in proto_remoteclass_id_mapping:
            raise TypeError(f"Class '{actual_cls.__name__}' is not registered as @egrpc.remoteclass. "
                            f"All subclasses of '{cls.__name__}' must be decorated with @egrpc.remoteclass.")

        instance_ref = get_ref_from_instance(actual_cls, obj, on_server)
        msg = getattr(proto_msg, param_name)
        msg.ref = instance_ref
        msg.type_id = proto_remoteclass_id_mapping[actual_cls]

    elif type_origin in (Union, UnionType):
        child_proto_msg = getattr(proto_msg, param_name)
        for i, th in enumerate(type_args):
            # TODO: consider what to do when obj matches multiple candidates
            if is_type_match(obj, th):
                set_proto_field(child_proto_msg, f"member{i}", th, obj, allow_subclass=allow_subclass, on_server=on_server)
                return
        raise TypeError(f"Type '{type(obj)}' of parameter '{param_name}' does not match any of {type_hint}.")

    elif type_origin in (tuple, Tuple):
        if len(type_args) == 2 and type_args[1] is Ellipsis:
            repeated_container = getattr(proto_msg, param_name).items
            set_proto_repeated_field(repeated_container, param_name, type_args[0], obj, allow_subclass=allow_subclass, on_server=on_server)
        else:
            child_proto_msg = getattr(proto_msg, param_name)
            for i, (th, o) in enumerate(zip(type_args, obj)):
                set_proto_field(child_proto_msg, f"item{i}", th, o, allow_subclass=allow_subclass, on_server=on_server)

    elif type_origin in (list, List, Sequence):
        repeated_container = getattr(proto_msg, param_name).elements
        set_proto_repeated_field(repeated_container, param_name, type_args[0], obj, allow_subclass=type_origin is Sequence, on_server=on_server)

    elif type_origin in (dict, Dict, Mapping):
        repeated_container_k = getattr(proto_msg, param_name).keys
        repeated_container_v = getattr(proto_msg, param_name).values
        # Mapping key type is not covariant
        set_proto_repeated_field(repeated_container_k, param_name, type_args[0], obj.keys(), allow_subclass=False, on_server=on_server)
        set_proto_repeated_field(repeated_container_v, param_name, type_args[1], obj.values(), allow_subclass=type_origin is Mapping, on_server=on_server)

    else:
        handler = get_handler_for_type(type_origin)
        if handler is not None:
            payload_obj = handler.pack(obj)
            set_proto_field(proto_msg, param_name, handler, payload_obj, allow_subclass=allow_subclass, on_server=on_server)
        else:
            raise TypeError(f"Type {type_origin} is not supported.")

def set_proto_repeated_field(repeated_container: Any, param_name: str, type_hint: TypeHint, objs: list[Any], allow_subclass: bool, on_server: bool) -> None:
    if type_hint in proto_primitive_type_mapping:
        # RepeatedScalarContainer
        repeated_container.extend(objs)
    else:
        # RepeatedCompositeContainer
        for o in objs:
            # https://stackoverflow.com/questions/57222346/how-to-get-type-contained-by-protobufs-repeatedcompositecontainer-or-repeatedsc
            child_proto_msg = repeated_container.add()
            WrapperMsg = type("WrapperMessage", (), {param_name: child_proto_msg})
            set_proto_field(WrapperMsg, param_name, type_hint, o, allow_subclass=allow_subclass, on_server=on_server)
