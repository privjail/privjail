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

from typing import TypeVar, Callable, Type, Any, ParamSpec
from types import ModuleType
from concurrent import futures
import traceback
import grpc # type: ignore[import-untyped]

from . import names
from .proto_interface import ProtoMsg

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

dynamic_pb2_grpc: ModuleType | None = None

def init_grpc(module: ModuleType) -> None:
    global dynamic_pb2_grpc
    dynamic_pb2_grpc = module

def get_grpc_module() -> ModuleType:
    global dynamic_pb2_grpc
    assert dynamic_pb2_grpc is not None
    return dynamic_pb2_grpc

HandlerType = Callable[[Any, ProtoMsg, Any], ProtoMsg]

proto_handlers: dict[str, dict[str, HandlerType]] = {}

def grpc_register_function(func: Callable[P, R], handler: HandlerType) -> None:
    proto_service_name = names.proto_function_service_name(func)
    proto_rpc_name = names.proto_function_rpc_name(func)

    # for debugging
    def wrapper(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
        try:
            return handler(self, proto_req, context)
        except:
            traceback.print_exc()
            raise

    global proto_handlers
    assert proto_service_name not in proto_handlers
    # proto_handlers[proto_service_name] = {proto_rpc_name: handler}
    proto_handlers[proto_service_name] = {proto_rpc_name: wrapper}

def grpc_register_method(cls: Type[T], method: Callable[P, R], handler: HandlerType) -> None:
    proto_service_name = names.proto_remoteclass_service_name(cls)
    proto_rpc_name = names.proto_method_rpc_name(cls, method)

    # for debugging
    def wrapper(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
        try:
            return handler(self, proto_req, context)
        except:
            traceback.print_exc()
            raise

    global proto_handlers
    if proto_service_name not in proto_handlers:
        proto_handlers[proto_service_name] = {}
    # proto_handlers[proto_service_name][proto_rpc_name] = handler
    proto_handlers[proto_service_name][proto_rpc_name] = wrapper

MAX_MESSAGE_LENGTH = 256 * 1024 * 1024  # 256MB

def init_server(port: int, host: str | None = None) -> Any:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1),
                         maximum_concurrent_rpcs=1,
                         options=(
                             ("grpc.so_reuseport", 0),
                             ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                             ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                         ))

    global proto_handlers
    for proto_service_name, handlers in proto_handlers.items():
        DynamicServicer = type(f"{proto_service_name}DynamicServicer",
                               (getattr(dynamic_pb2_grpc, f"{proto_service_name}Servicer"),),
                               handlers)

        add_service_fn = getattr(dynamic_pb2_grpc, f"add_{proto_service_name}Servicer_to_server")
        add_service_fn(DynamicServicer(), server)

    bind_host = host if host is not None else "[::]"
    server.add_insecure_port(f"{bind_host}:{port}")

    return server

ChannelType = Any

client_channel: ChannelType | None = None

def init_client(hostname: str, port: int) -> None:
    global client_channel
    client_channel = grpc.insecure_channel(
        f"{hostname}:{port}",
        options=(
            ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
        )
    )

def del_client() -> None:
    global client_channel
    client_channel = None

def get_client_channel() -> ChannelType:
    global client_channel
    assert client_channel is not None
    return client_channel

def remote_connected() -> bool:
    return client_channel is not None

def grpc_function_call(func: Callable[P, R], proto_req: ProtoMsg) -> ProtoMsg:
    proto_service_name = names.proto_function_service_name(func)
    proto_rpc_name = names.proto_function_rpc_name(func)

    channel = get_client_channel()
    stub = getattr(get_grpc_module(), f"{proto_service_name}Stub")(channel)
    proto_res = getattr(stub, proto_rpc_name)(proto_req)

    return proto_res

def grpc_method_call(cls: Type[T], method: Callable[P, R], proto_req: ProtoMsg) -> ProtoMsg:
    proto_service_name = names.proto_remoteclass_service_name(cls)
    proto_rpc_name = names.proto_method_rpc_name(cls, method)

    channel = get_client_channel()
    stub = getattr(get_grpc_module(), f"{proto_service_name}Stub")(channel)
    proto_res = getattr(stub, proto_rpc_name)(proto_req)

    return proto_res
