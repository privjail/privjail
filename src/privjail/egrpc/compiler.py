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
from typing import get_args, Union, List, Tuple, Dict, Callable, Any, TypeVar, Type, ParamSpec
from types import UnionType, ModuleType
from collections.abc import Sequence, Mapping
import os
import sys
import tempfile
import importlib.util
from grpc_tools import protoc # type: ignore[import-untyped]

# TODO: make egrpc independent of numpy
import numpy as _np

from . import names
from .util import get_function_typed_params, get_function_return_type, get_class_typed_members, get_method_typed_params, get_method_return_type, TypeHint, my_get_origin
from .instance_ref import InstanceRefType

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

proto_primitive_type_mapping = {
    str          : "string",
    int          : "int64",
    float        : "double",
    bool         : "bool",
    type(None)   : "bool",
    _np.integer  : "int64",
    _np.floating : "double",
}

proto_dataclass_type_mapping: dict[Any, str] = {}
proto_remoteclass_type_mapping: dict[Any, str] = {}

proto_header = """syntax = "proto3";
"""

proto_content = ""

def indent_str(depth: int) -> str:
    return " " * depth * 2

def is_subclass(type_hint: TypeHint, type_mapping: dict[Any, str]) -> bool:
    return isinstance(type_hint, type) and issubclass(type_hint, tuple(type_mapping.keys()))

def subclasses(type_hint: TypeHint, type_mapping: dict[Any, str]) -> dict[TypeHint, str]:
    if isinstance(type_hint, type):
        return {th: proto_type for th, proto_type in type_mapping.items() if issubclass(th, type_hint)}
    else:
        return {}

def gen_proto_field_def(index          : int,
                        param_name     : str,
                        type_hint      : TypeHint,
                        allow_subclass : bool = True,
                        repeated       : bool = False,
                        depth          : int = 0) -> tuple[list[str], list[str], int]:
    type_origin = my_get_origin(type_hint)
    type_args = get_args(type_hint)

    type_origin = type_hint if type_origin is None else type_origin

    repeated_str = "repeated " if repeated else ""

    proto_fields = []
    proto_defs = []

    if type_origin in proto_primitive_type_mapping:
        proto_type = proto_primitive_type_mapping[type_origin]
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{proto_type} {param_name} = {index + 1};")
        index += 1

    elif is_subclass(type_origin, proto_dataclass_type_mapping):
        # TODO: move subtype handling to `compile_dataclass()`?
        proto_types = subclasses(type_origin, proto_dataclass_type_mapping)
        if not allow_subclass or len(proto_types) == 0:
            proto_type = proto_dataclass_type_mapping[type_origin]
            proto_fields.append(f"{indent_str(depth)}{repeated_str}{proto_type} {param_name} = {index + 1};")
            index += 1
        else:
            msgname = f"{param_name.capitalize()}SubclassMessage"
            child_type_hints = {f"class{i}": th for i, th in enumerate(proto_types.keys())}
            proto_defs += gen_proto_msg_def(msgname, child_type_hints, allow_subclass=False, oneof=True)
            proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
            index += 1

    elif type_origin in proto_remoteclass_type_mapping:
        proto_type = proto_remoteclass_type_mapping[type_origin]
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{proto_type} {param_name} = {index + 1};")
        index += 1

    elif type_origin in (Union, UnionType):
        msgname = f"{param_name.capitalize()}UnionMessage"
        child_type_hints = {f"member{i}": th for i, th in enumerate(type_args)}
        proto_defs += gen_proto_msg_def(msgname, child_type_hints, allow_subclass=allow_subclass, oneof=True)
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
        index += 1

    elif type_origin in (tuple, Tuple):
        msgname = f"{param_name.capitalize()}TupleMessage"
        child_type_hints = {f"item{i}": th for i, th in enumerate(type_args)}
        proto_defs += gen_proto_msg_def(msgname, child_type_hints, allow_subclass=allow_subclass)
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
        index += 1

    elif type_origin in (list, List, Sequence):
        msgname = f"{param_name.capitalize()}ListMessage"
        proto_defs += gen_proto_msg_def(msgname, {"elements": type_args[0]}, allow_subclass=type_origin is Sequence, repeated=True)
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
        index += 1

    elif type_origin in (dict, Dict, Mapping):
        msgname = f"{param_name.capitalize()}DictMessage"
        proto_defs += gen_proto_msg_def(msgname, {"keys": type_args[0], "values": type_args[1]}, allow_subclass=type_origin is Mapping, repeated=True)
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
        index += 1

    else:
        raise TypeError(f"Type {type_origin} is not supported.")

    return proto_fields, proto_defs, index

def gen_proto_msg_def(msgname        : str,
                      typed_params   : dict[str, TypeHint],
                      allow_subclass : bool = True,
                      repeated       : bool = False,
                      oneof          : bool = False,
                      depth          : int = 0) -> list[str]:
    index = 0
    proto_defs = []
    proto_inner_defs = []

    proto_defs.append(f"{indent_str(depth)}message {msgname} {{")

    if oneof:
        proto_defs.append(f"{indent_str(depth + 1)}oneof wrapper {{")

    for param_name, type_hint in typed_params.items():
        next_depth = depth + 2 if oneof else depth + 1
        pf, pd, index = gen_proto_field_def(index, param_name, type_hint, allow_subclass=allow_subclass, repeated=repeated, depth=next_depth)
        proto_defs += pf
        proto_inner_defs += pd

    if oneof:
        proto_defs.append(f"{indent_str(depth + 1)}}}")

    proto_defs += [indent_str(depth + 1) + line for line in proto_inner_defs]

    proto_defs.append(f"{indent_str(depth)}}}")

    return proto_defs

deferred_compilation: list[Callable[[], None]] = []

def defer(func: Callable[[], None]) -> None:
    global deferred_compilation
    deferred_compilation.append(func)

def do_deferred() -> None:
    global deferred_compilation
    for func in deferred_compilation:
        func()
    deferred_compilation = []

def compile_function(func: Callable[P, R]) -> None:
    def do_compile() -> None:
        typed_params = get_function_typed_params(func)
        return_type = get_function_return_type(func)

        proto_service_name = names.proto_function_service_name(func)
        proto_rpc_name = names.proto_function_rpc_name(func)
        proto_req_name = names.proto_function_req_name(func)
        proto_res_name = names.proto_function_res_name(func)

        proto_req_def = gen_proto_msg_def(proto_req_name, typed_params)
        proto_res_def = gen_proto_msg_def(proto_res_name, {"return": return_type})

        global proto_content
        proto_content += f"""
service {proto_service_name} {{
  rpc {proto_rpc_name} ({proto_req_name}) returns ({proto_res_name});
}}

{chr(10).join(proto_req_def)}

{chr(10).join(proto_res_def)}
"""

    defer(do_compile)

def compile_dataclass(cls: Type[T]) -> None:
    def do_compile() -> None:
        proto_msgname = names.proto_dataclass_name(cls)

        proto_def = gen_proto_msg_def(proto_msgname, get_class_typed_members(cls))

        global proto_content
        proto_content += "\n"
        proto_content += "\n".join(proto_def)
        proto_content += "\n"

        global proto_dataclass_type_mapping
        proto_dataclass_type_mapping[cls] = proto_msgname

    defer(do_compile)

proto_remoteclass_rpc_defs: dict[Type[Any], list[str]] = {}
proto_remoteclass_msg_defs: dict[Type[Any], list[str]] = {}

def add_remoteclass_method_def(cls: Type[T], proto_rpc_def: str, proto_req_def: list[str], proto_res_def: list[str]) -> None:
    global proto_remoteclass_rpc_defs, proto_remoteclass_msg_defs

    if cls not in proto_remoteclass_rpc_defs:
        proto_remoteclass_rpc_defs[cls] = []
    if cls not in proto_remoteclass_msg_defs:
        proto_remoteclass_msg_defs[cls] = []

    proto_remoteclass_rpc_defs[cls].append(proto_rpc_def)
    proto_remoteclass_msg_defs[cls] += [""] + proto_req_def
    proto_remoteclass_msg_defs[cls] += [""] + proto_res_def

def define_remoteclass_instance_ref(cls: Type[T]) -> None:
    global proto_remoteclass_type_mapping

    if cls not in proto_remoteclass_type_mapping:
        proto_instance_ref_name = names.proto_instance_ref_name(cls)

        proto_remoteclass_type_mapping[cls] = proto_instance_ref_name

        global proto_remoteclass_msg_defs

        if cls not in proto_remoteclass_msg_defs:
            proto_remoteclass_msg_defs[cls] = []

        proto_instance_def = gen_proto_msg_def(proto_instance_ref_name, {"ref": InstanceRefType})
        proto_remoteclass_msg_defs[cls] += [""] + proto_instance_def

def compile_remoteclass(cls: Type[T]) -> None:
    define_remoteclass_instance_ref(cls)

    def do_compile() -> None:
        proto_service_name = names.proto_remoteclass_service_name(cls)

        global proto_content
        proto_content += f"""
service {proto_service_name} {{
{chr(10).join(proto_remoteclass_rpc_defs[cls])}
}}
{chr(10).join(proto_remoteclass_msg_defs[cls])}
"""

        del proto_remoteclass_rpc_defs[cls]
        del proto_remoteclass_msg_defs[cls]

    defer(do_compile)

def compile_remoteclass_init(cls: Type[T], init_method: Callable[P, R]) -> None:
    define_remoteclass_instance_ref(cls)

    def do_compile() -> None:
        typed_params = get_method_typed_params(cls, init_method)

        proto_rpc_name = names.proto_method_rpc_name(cls, init_method)
        proto_req_name = names.proto_method_req_name(cls, init_method)
        proto_res_name = names.proto_method_res_name(cls, init_method)

        proto_rpc_def = f"  rpc {proto_rpc_name} ({proto_req_name}) returns ({proto_res_name});"
        proto_req_def = gen_proto_msg_def(proto_req_name, dict(list(typed_params.items())[1:]))
        proto_res_def = gen_proto_msg_def(proto_res_name, {"return": InstanceRefType})

        add_remoteclass_method_def(cls, proto_rpc_def, proto_req_def, proto_res_def)

    defer(do_compile)

def compile_remoteclass_del(cls: Type[T], del_method: Callable[P, R]) -> None:
    define_remoteclass_instance_ref(cls)

    def do_compile() -> None:
        typed_params = get_method_typed_params(cls, del_method)

        proto_rpc_name = names.proto_method_rpc_name(cls, del_method)
        proto_req_name = names.proto_method_req_name(cls, del_method)
        proto_res_name = names.proto_method_res_name(cls, del_method)

        proto_rpc_def = f"  rpc {proto_rpc_name} ({proto_req_name}) returns ({proto_res_name});"
        proto_req_def = gen_proto_msg_def(proto_req_name, typed_params)
        proto_res_def = gen_proto_msg_def(proto_res_name, {})

        add_remoteclass_method_def(cls, proto_rpc_def, proto_req_def, proto_res_def)

    defer(do_compile)

def compile_remoteclass_method(cls: Type[T], method: Callable[P, R]) -> None:
    define_remoteclass_instance_ref(cls)

    def do_compile() -> None:
        typed_params = get_method_typed_params(cls, method)
        return_type = get_method_return_type(cls, method)

        proto_rpc_name = names.proto_method_rpc_name(cls, method)
        proto_req_name = names.proto_method_req_name(cls, method)
        proto_res_name = names.proto_method_res_name(cls, method)

        proto_rpc_def = f"  rpc {proto_rpc_name} ({proto_req_name}) returns ({proto_res_name});"
        proto_req_def = gen_proto_msg_def(proto_req_name, typed_params)
        proto_res_def = gen_proto_msg_def(proto_res_name, {"return": return_type})

        add_remoteclass_method_def(cls, proto_rpc_def, proto_req_def, proto_res_def)

    defer(do_compile)

def compile_proto() -> tuple[ModuleType, ModuleType]:
    do_deferred()

    with tempfile.TemporaryDirectory(prefix="egrpc_") as tempdir:
        proto_file = os.path.join(tempdir, "dynamic.proto")

        with open(proto_file, "w") as f:
            f.write(proto_file_content())

        protoc.main(
            [
                "grpc_tools.protoc",
                f"--proto_path={tempdir}",
                f"--python_out={tempdir}",
                f"--grpc_python_out={tempdir}",
                proto_file,
            ]
        )

        def import_dynamic_module(module_name: str, filename: str) -> ModuleType:
            filepath = os.path.join(tempdir, filename)
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        dynamic_pb2 = import_dynamic_module("dynamic_pb2", "dynamic_pb2.py")
        dynamic_pb2_grpc = import_dynamic_module("dynamic_pb2_grpc", "dynamic_pb2_grpc.py")

        return dynamic_pb2, dynamic_pb2_grpc

def proto_file_content() -> str:
    do_deferred()
    return proto_header + proto_content
