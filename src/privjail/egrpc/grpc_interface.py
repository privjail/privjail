from concurrent import futures
import grpc

from . import names

dynamic_pb2_grpc = None

def init_grpc(module):
    global dynamic_pb2_grpc
    dynamic_pb2_grpc = module

def get_grpc_module():
    global dynamic_pb2_grpc
    assert dynamic_pb2_grpc is not None
    return dynamic_pb2_grpc

proto_handlers = {}

def grpc_register_service(proto_service_name, handlers):
    global proto_handlers
    proto_handlers[proto_service_name] = handlers

def init_server(port):
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

client_channel = None

def init_client(hostname, port):
    global client_channel
    client_channel = grpc.insecure_channel(f"{hostname}:{port}")

def get_client_channel():
    global client_channel
    assert client_channel is not None
    return client_channel

def grpc_function_call(func, proto_req):
    proto_service_name = names.proto_function_service_name(func)
    proto_rpc_name = names.proto_function_rpc_name(func)

    channel = get_client_channel()
    stub = getattr(get_grpc_module(), f"{proto_service_name}Stub")(channel)
    proto_res = getattr(stub, proto_rpc_name)(proto_req)

    return proto_res

def grpc_method_call(cls, method, proto_req):
    proto_service_name = names.proto_remoteclass_service_name(cls)
    proto_rpc_name = names.proto_method_rpc_name(cls, method)

    channel = get_client_channel()
    stub = getattr(get_grpc_module(), f"{proto_service_name}Stub")(channel)
    proto_res = getattr(stub, proto_rpc_name)(proto_req)

    return proto_res
