from typing import get_origin, get_args, Union, List, Tuple, Dict
import types
import sys
import importlib.util
from grpc_tools import protoc

from . import names
from .util import get_function_typed_params, get_function_return_type, get_class_typed_members, get_method_self_name, get_method_typed_params, get_method_return_type

InstanceRef = int

proto_primitive_type_mapping = {
    str        : "string",
    int        : "int64",
    float      : "float",
    bool       : "bool",
    type(None) : "bool",
}

proto_dataclass_type_mapping = {}
proto_remoteclass_type_mapping = {}

proto_header = """syntax = "proto3";
"""

proto_content = ""

def indent_str(depth):
    return " " * depth * 2

def gen_proto_field_def(index, param_name, type_hint, repeated=False, depth=0):
    type_origin = get_origin(type_hint)
    type_args = get_args(type_hint)

    repeated_str = "repeated " if repeated else ""

    proto_fields = []
    proto_defs = []

    if type_origin is None:
        proto_type_mapping = {**proto_primitive_type_mapping,
                              **proto_dataclass_type_mapping,
                              **proto_remoteclass_type_mapping}
        proto_type = proto_type_mapping[type_hint]
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{proto_type} {param_name} = {index + 1};")
        index += 1

    elif type_origin in (Union, types.UnionType):
        msgname = f"{param_name.capitalize()}UnionMessage"
        child_type_hints = {f"member{i}": th for i, th in enumerate(type_args)}
        proto_defs += gen_proto_msg_def(msgname, child_type_hints, oneof=True)
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
        index += 1

    elif type_origin in (tuple, Tuple):
        msgname = f"{param_name.capitalize()}TupleMessage"
        child_type_hints = {f"item{i}": th for i, th in enumerate(type_args)}
        proto_defs += gen_proto_msg_def(msgname, child_type_hints)
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
        index += 1

    elif type_origin in (list, List):
        msgname = f"{param_name.capitalize()}ListMessage"
        proto_defs += gen_proto_msg_def(msgname, {"elements": type_args[0]}, repeated=True)
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
        index += 1

    elif type_origin in (dict, Dict):
        msgname = f"{param_name.capitalize()}DictMessage"
        proto_defs += gen_proto_msg_def(msgname, {"keys": type_args[0], "values": type_args[1]}, repeated=True)
        proto_fields.append(f"{indent_str(depth)}{repeated_str}{msgname} {param_name} = {index + 1};")
        index += 1

    else:
        raise Exception

    return proto_fields, proto_defs, index

def gen_proto_msg_def(msgname, typed_params, repeated=False, oneof=False, depth=0):
    index = 0
    proto_defs = []
    proto_inner_defs = []

    proto_defs.append(f"{indent_str(depth)}message {msgname} {{")

    if oneof:
        proto_defs.append(f"{indent_str(depth + 1)}oneof wrapper {{")

    for param_name, type_hint in typed_params.items():
        next_depth = depth + 2 if oneof else depth + 1
        pf, pd, index = gen_proto_field_def(index, param_name, type_hint, repeated=repeated, depth=next_depth)
        proto_defs += pf
        proto_inner_defs += pd

    if oneof:
        proto_defs.append(f"{indent_str(depth + 1)}}}")

    proto_defs += [indent_str(depth + 1) + line for line in proto_inner_defs]

    proto_defs.append(f"{indent_str(depth)}}}")

    return proto_defs

def compile_function(func):
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

def compile_dataclass(cls):
    proto_msgname = names.proto_dataclass_name(cls)

    proto_def = gen_proto_msg_def(proto_msgname, get_class_typed_members(cls))

    global proto_content
    proto_content += "\n"
    proto_content += "\n".join(proto_def)
    proto_content += "\n"

    global proto_dataclass_type_mapping
    proto_dataclass_type_mapping[cls] = proto_msgname

def compile_remoteclass(cls, methods):
    global proto_remoteclass_type_mapping
    proto_remoteclass_type_mapping[cls] = proto_primitive_type_mapping[InstanceRef]

    proto_service_name = names.proto_remoteclass_service_name(cls)

    proto_rpc_def = []
    proto_msg_def = []

    proto_instance_def = gen_proto_msg_def(names.proto_instance_ref_name(cls), {"instance_ref": InstanceRef})
    proto_msg_def += [""] + proto_instance_def

    compile_remoteclass_init(cls, proto_rpc_def, proto_msg_def)
    compile_remoteclass_del(cls, proto_rpc_def, proto_msg_def)

    for method_name, method in methods.items():
        compile_remoteclass_method(cls, method, proto_rpc_def, proto_msg_def)

    global proto_content
    proto_content += f"""
service {proto_service_name} {{
{chr(10).join(proto_rpc_def)}
}}
{chr(10).join(proto_msg_def)}
"""

def compile_remoteclass_init(cls, proto_rpc_def, proto_msg_def):
    typed_params = get_method_typed_params(cls, cls.__init__)

    proto_rpc_name = names.proto_method_rpc_name(cls, cls.__init__)
    proto_req_name = names.proto_method_req_name(cls, cls.__init__)
    proto_res_name = names.proto_method_res_name(cls, cls.__init__)

    proto_req_def = gen_proto_msg_def(proto_req_name, typed_params)
    proto_res_def = gen_proto_msg_def(proto_res_name, {"return": InstanceRef})

    proto_rpc_def.append(f"  rpc {proto_rpc_name} ({proto_req_name}) returns ({proto_res_name});")
    proto_msg_def += [""] + proto_req_def + [""] + proto_res_def

def compile_remoteclass_del(cls, proto_rpc_def, proto_msg_def):
    self_name = get_method_self_name(cls, cls.__del__)

    proto_rpc_name = names.proto_method_rpc_name(cls, cls.__del__)
    proto_req_name = names.proto_method_req_name(cls, cls.__del__)
    proto_res_name = names.proto_method_res_name(cls, cls.__del__)

    proto_req_def = gen_proto_msg_def(proto_req_name, {self_name: InstanceRef})
    proto_res_def = gen_proto_msg_def(proto_res_name, {})

    proto_rpc_def.append(f"  rpc {proto_rpc_name} ({proto_req_name}) returns ({proto_res_name});")
    proto_msg_def += [""] + proto_req_def + [""] + proto_res_def

def compile_remoteclass_method(cls, method, proto_rpc_def, proto_msg_def):
    self_name = get_method_self_name(cls, method)
    typed_params = get_method_typed_params(cls, method)
    return_type = get_method_return_type(cls, method)

    proto_rpc_name = names.proto_method_rpc_name(cls, method)
    proto_req_name = names.proto_method_req_name(cls, method)
    proto_res_name = names.proto_method_res_name(cls, method)

    proto_req_def = gen_proto_msg_def(proto_req_name, {self_name: InstanceRef, **typed_params})
    proto_res_def = gen_proto_msg_def(proto_res_name, {"return": return_type})

    proto_rpc_def.append(f"  rpc {proto_rpc_name} ({proto_req_name}) returns ({proto_res_name});")
    proto_msg_def += [""] + proto_req_def + [""] + proto_res_def

def compile_proto():
    proto_file = "dynamic.proto"
    with open(proto_file, "w") as f:
        f.write(proto_header + proto_content)

    protoc.main(
        [
            "grpc_tools.protoc",
            "--proto_path=.",
            "--python_out=.",
            "--grpc_python_out=.",
            proto_file,
        ]
    )

    spec = importlib.util.spec_from_file_location("dynamic_pb2", "./dynamic_pb2.py")
    dynamic_pb2 = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_pb2"] = dynamic_pb2
    spec.loader.exec_module(dynamic_pb2)

    spec_grpc = importlib.util.spec_from_file_location("dynamic_pb2_grpc", "./dynamic_pb2_grpc.py")
    dynamic_pb2_grpc = importlib.util.module_from_spec(spec_grpc)
    sys.modules["dynamic_pb2_grpc"] = dynamic_pb2_grpc
    spec_grpc.loader.exec_module(dynamic_pb2_grpc)

    return dynamic_pb2, dynamic_pb2_grpc
