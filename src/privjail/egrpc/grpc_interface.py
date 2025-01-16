from typing import TypeVar, Callable, Type, Any, ParamSpec
from types import ModuleType
from concurrent import futures
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

    global proto_handlers
    assert proto_service_name not in proto_handlers
    proto_handlers[proto_service_name] = {proto_rpc_name: handler}

def grpc_register_method(cls: Type[T], method: Callable[P, R], handler: HandlerType) -> None:
    proto_service_name = names.proto_remoteclass_service_name(cls)
    proto_rpc_name = names.proto_method_rpc_name(cls, method)

    global proto_handlers
    if proto_service_name not in proto_handlers:
        proto_handlers[proto_service_name] = {}
    proto_handlers[proto_service_name][proto_rpc_name] = handler

def init_server(port: int) -> Any:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), maximum_concurrent_rpcs=1)

    global proto_handlers
    for proto_service_name, handlers in proto_handlers.items():
        DynamicServicer = type(f"{proto_service_name}DynamicServicer",
                               (getattr(dynamic_pb2_grpc, f"{proto_service_name}Servicer"),),
                               handlers)

        add_service_fn = getattr(dynamic_pb2_grpc, f"add_{proto_service_name}Servicer_to_server")
        add_service_fn(DynamicServicer(), server)

    server.add_insecure_port(f"[::]:{port}")

    return server

ChannelType = Any

client_channel: ChannelType | None = None

def init_client(hostname: str, port: int) -> None:
    global client_channel
    client_channel = grpc.insecure_channel(f"{hostname}:{port}")

def get_client_channel() -> ChannelType:
    global client_channel
    assert client_channel is not None
    return client_channel

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
