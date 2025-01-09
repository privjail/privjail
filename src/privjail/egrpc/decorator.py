import sys
import functools
import dataclasses
import weakref

from . import names
from .util import egrpc_mode
from .compiler import compile_function, compile_dataclass, compile_remoteclass
from .proto_interface import pack_proto_function_request, pack_proto_function_response, unpack_proto_function_request, unpack_proto_function_response, pack_proto_method_request, pack_proto_method_response, unpack_proto_method_request, unpack_proto_method_response
from .grpc_interface import grpc_register_service, grpc_function_call, grpc_method_call

def function_decorator(func):
    compile_function(func)

    if egrpc_mode == "server":
        def function_handler(self, proto_req, context):
            args = unpack_proto_function_request(func, proto_req)
            result = func(**args)
            return pack_proto_function_response(func, result)

        proto_rpc_name = names.proto_function_rpc_name(func)
        proto_service_name = names.proto_function_service_name(func)
        grpc_register_service(proto_service_name, {proto_rpc_name: function_handler})

        return func

    elif egrpc_mode == "client":
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            proto_req = pack_proto_function_request(func, *args, **kwargs)
            proto_res = grpc_function_call(func, proto_req)
            return unpack_proto_function_response(func, proto_res)

        return wrapper

def dataclass_decorator(cls):
    datacls = dataclasses.dataclass(cls)
    compile_dataclass(datacls)
    return datacls

def method_decorator(method):
    method.__egrpc_enabled = True
    return method

def remoteclass_decorator(cls):
    methods = {k: v for k, v in cls.__dict__.items() if hasattr(v, "__egrpc_enabled")}

    compile_remoteclass(cls, methods)

    cls.__instance_count = 0
    cls.__instance_map = {}

    if egrpc_mode == "server":
        handlers = {}

        def init_handler(self, proto_req, context):
            args = unpack_proto_method_request(cls, cls.__init__, proto_req)
            obj = cls(**args)
            return pack_proto_method_response(cls, cls.__init__, obj)

        proto_rpc_name = names.proto_method_rpc_name(cls, cls.__init__)
        handlers[proto_rpc_name] = init_handler

        def del_handler(self, proto_req, context):
            args = unpack_proto_method_request(cls, cls.__del__, proto_req)
            obj = list(args.values())[0]
            del cls.__instance_map[obj.__instance_ref]
            assert sys.getrefcount(obj) == 3
            return pack_proto_method_response(cls, cls.__del__, None)

        proto_rpc_name = names.proto_method_rpc_name(cls, cls.__del__)
        handlers[proto_rpc_name] = del_handler

        for method_name, method in methods.items():
            def method_handler(self, proto_req, context, method=method):
                args = unpack_proto_method_request(cls, method, proto_req)
                result = method(**args)
                return pack_proto_method_response(cls, method, result)

            proto_rpc_name = names.proto_method_rpc_name(cls, method)
            handlers[proto_rpc_name] = method_handler

        proto_service_name = names.proto_remoteclass_service_name(cls)
        grpc_register_service(proto_service_name, handlers)

        return cls

    elif egrpc_mode == "client":
        @functools.wraps(cls.__init__)
        def init_wrapper(*args, **kwargs):
            proto_req = pack_proto_method_request(cls, cls.__init__, *args, **kwargs)
            proto_res = grpc_method_call(cls, cls.__init__, proto_req)
            instance_ref = unpack_proto_method_response(cls, cls.__init__, proto_res)

            self = args[0]

            assert instance_ref not in cls.__instance_map
            cls.__instance_map[instance_ref] = weakref.ref(self)

            self.__instance_ref = instance_ref

        cls.__init__ = init_wrapper

        @functools.wraps(cls.__del__)
        def del_wrapper(self):
            proto_req = pack_proto_method_request(cls, cls.__del__, self)
            grpc_method_call(cls, cls.__del__, proto_req)

        cls.__del__ = del_wrapper

        for method_name, method in methods.items():
            @functools.wraps(method)
            def wrapper(*args, method=method, **kwargs):
                proto_req = pack_proto_method_request(cls, method, *args, **kwargs)
                proto_res = grpc_method_call(cls, method, proto_req)
                return unpack_proto_method_response(cls, method, proto_res)

            setattr(cls, method_name, wrapper)

        return cls
