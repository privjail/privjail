import os

from .compiler import compile_proto
from .proto_interface import init_proto
from .grpc_interface import init_grpc, init_server, init_client

def init() -> None:
    dynamic_pb2, dynamic_pb2_grpc = compile_proto()
    init_proto(dynamic_pb2)
    init_grpc(dynamic_pb2_grpc)

def serve(port: int) -> None:
    init()
    server = init_server(port)
    print(f"Server started on port {port} (pid = {os.getpid()}).")
    server.start()
    server.wait_for_termination()

def connect(hostname: str, port: int) -> None:
    init()
    init_client(hostname, port)
    print(f"Client connected to server {hostname}:{port} (pid = {os.getpid()}).")
