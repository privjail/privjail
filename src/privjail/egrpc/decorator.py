import functools
import dataclasses

from .util import egrpc_mode
from .compiler import compile_function, compile_dataclass, compile_remoteclass
from .proto_interface import pack_proto_function_request, pack_proto_function_response, unpack_proto_function_request, unpack_proto_function_response, pack_proto_method_request, pack_proto_method_response, unpack_proto_method_request, unpack_proto_method_response
from .grpc_interface import grpc_register_function, grpc_register_method, grpc_function_call, grpc_method_call
from .instance_ref import init_remoteclass, assign_ref_to_instance, del_instance

def function_decorator(func):
    compile_function(func)

    if egrpc_mode == "server":
        def function_handler(self, proto_req, context):
            args = unpack_proto_function_request(func, proto_req)
            result = func(**args)
            return pack_proto_function_response(func, result)

        grpc_register_function(func, function_handler)

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
    init_remoteclass(cls)

    methods = {k: v for k, v in cls.__dict__.items() if hasattr(v, "__egrpc_enabled")}

    compile_remoteclass(cls, methods)

    if egrpc_mode == "server":
        for method_name, method in methods.items():
            if method_name == "__init__":
                def init_handler(self, proto_req, context):
                    args = unpack_proto_method_request(cls, cls.__init__, proto_req)
                    obj = cls(**args)
                    return pack_proto_method_response(cls, cls.__init__, obj)

                grpc_register_method(cls, cls.__init__, init_handler)

            else:
                def method_handler(self, proto_req, context, method=method):
                    args = unpack_proto_method_request(cls, method, proto_req)
                    result = method(**args)
                    return pack_proto_method_response(cls, method, result)

                grpc_register_method(cls, method, method_handler)

        def del_handler(self, proto_req, context):
            args = unpack_proto_method_request(cls, cls.__del__, proto_req)
            del_instance(cls, list(args.values())[0])
            return pack_proto_method_response(cls, cls.__del__, None)

        grpc_register_method(cls, cls.__del__, del_handler)

        return cls

    elif egrpc_mode == "client":
        for method_name, method in methods.items():
            if method_name == "__init__":
                @functools.wraps(cls.__init__)
                def init_wrapper(*args, **kwargs):
                    proto_req = pack_proto_method_request(cls, cls.__init__, *args, **kwargs)
                    proto_res = grpc_method_call(cls, cls.__init__, proto_req)
                    instance_ref = unpack_proto_method_response(cls, cls.__init__, proto_res)
                    assign_ref_to_instance(cls, args[0], instance_ref)

                cls.__init__ = init_wrapper

            else:
                @functools.wraps(method)
                def wrapper(*args, method=method, **kwargs):
                    proto_req = pack_proto_method_request(cls, method, *args, **kwargs)
                    proto_res = grpc_method_call(cls, method, proto_req)
                    return unpack_proto_method_response(cls, method, proto_res)

                setattr(cls, method_name, wrapper)

        @functools.wraps(cls.__del__)
        def del_wrapper(self):
            proto_req = pack_proto_method_request(cls, cls.__del__, self)
            grpc_method_call(cls, cls.__del__, proto_req)

        cls.__del__ = del_wrapper

        return cls
