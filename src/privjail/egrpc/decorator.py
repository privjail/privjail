from typing import TypeVar, Callable, Type, Any, cast, ParamSpec
import functools
import dataclasses

from .util import egrpc_mode
from .compiler import compile_function, compile_dataclass, compile_remoteclass
from .proto_interface import pack_proto_function_request, pack_proto_function_response, unpack_proto_function_request, unpack_proto_function_response, pack_proto_method_request, pack_proto_method_response, unpack_proto_method_request, unpack_proto_method_response, ProtoMsg
from .grpc_interface import grpc_register_function, grpc_register_method, grpc_function_call, grpc_method_call
from .instance_ref import init_remoteclass, assign_ref_to_instance, del_instance

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

def function_decorator(func: Callable[P, R]) -> Callable[P, R]:
    if egrpc_mode == "server":
        compile_function(func)

        def function_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
            args = unpack_proto_function_request(func, proto_req)
            result = func(**args) # type: ignore[call-arg]
            return pack_proto_function_response(func, result)

        grpc_register_function(func, function_handler)

        return func

    elif egrpc_mode == "client":
        compile_function(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            proto_req = pack_proto_function_request(func, *args, **kwargs)
            proto_res = grpc_function_call(func, proto_req)
            return unpack_proto_function_response(func, proto_res)

        return cast(Callable[P, R], wrapper)

    else:
        return func

def dataclass_decorator(cls: Type[T]) -> Type[T]:
    datacls = dataclasses.dataclass(cls)
    if egrpc_mode in ["server", "client"]:
        compile_dataclass(datacls)
    return datacls

def method_decorator(method: Callable[P, R]) -> Callable[P, R]:
    if egrpc_mode in ["server", "client"]:
        setattr(method, "__egrpc_enabled", True)
    return method

def remoteclass_decorator(cls: Type[T]) -> Type[T]:
    if egrpc_mode == "server":
        init_remoteclass(cls)
        methods = {k: v for k, v in cls.__dict__.items() if hasattr(v, "__egrpc_enabled")}
        compile_remoteclass(cls, methods)

        for method_name, method in methods.items():
            if method_name == "__init__":
                init_method = getattr(cls, "__init__")

                def init_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
                    args = unpack_proto_method_request(cls, init_method, proto_req)
                    obj = cls(**args)
                    return pack_proto_method_response(cls, init_method, obj)

                grpc_register_method(cls, init_method, init_handler)

            else:
                def method_handler(self: Any, proto_req: ProtoMsg, context: Any, method: Callable[P, R] = method) -> ProtoMsg:
                    args = unpack_proto_method_request(cls, method, proto_req)
                    result = method(**args) # type: ignore[call-arg]
                    return pack_proto_method_response(cls, method, result)

                grpc_register_method(cls, method, method_handler)

        del_method = getattr(cls, "__del__")

        def del_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
            args = unpack_proto_method_request(cls, del_method, proto_req)
            del_instance(cls, list(args.values())[0])
            return pack_proto_method_response(cls, del_method, None)

        grpc_register_method(cls, del_method, del_handler)

        return cls

    elif egrpc_mode == "client":
        init_remoteclass(cls)
        methods = {k: v for k, v in cls.__dict__.items() if hasattr(v, "__egrpc_enabled")}
        compile_remoteclass(cls, methods)

        for method_name, method in methods.items():
            if method_name == "__init__":
                init_method = getattr(cls, "__init__")

                @functools.wraps(init_method)
                def init_wrapper(*args: Any, **kwargs: Any) -> None:
                    proto_req = pack_proto_method_request(cls, init_method, *args, **kwargs)
                    proto_res = grpc_method_call(cls, init_method, proto_req)
                    instance_ref = unpack_proto_method_response(cls, init_method, proto_res)
                    assign_ref_to_instance(cls, args[0], instance_ref)

                setattr(cls, "__init__", init_wrapper)

            else:
                @functools.wraps(method)
                def method_wrapper(*args: Any, method: Callable[P, R] = method, **kwargs: Any) -> Any:
                    proto_req = pack_proto_method_request(cls, method, *args, **kwargs)
                    proto_res = grpc_method_call(cls, method, proto_req)
                    return unpack_proto_method_response(cls, method, proto_res)

                setattr(cls, method_name, method_wrapper)

        del_method = getattr(cls, "__del__")

        @functools.wraps(del_method)
        def del_wrapper(self: Any) -> None:
            proto_req = pack_proto_method_request(cls, del_method, self)
            grpc_method_call(cls, del_method, proto_req)

        setattr(cls, "__del__", del_wrapper)

        return cls

    else:
        return cls
