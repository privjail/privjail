from types import MethodType
from typing import TypeVar, Callable, Type, Any, cast, ParamSpec, Generic
import functools
import dataclasses

from .compiler import compile_function, compile_dataclass, compile_remoteclass, compile_remoteclass_init, compile_remoteclass_method, compile_remoteclass_del
from .proto_interface import pack_proto_function_request, pack_proto_function_response, unpack_proto_function_request, unpack_proto_function_response, pack_proto_method_request, pack_proto_method_response, unpack_proto_method_request, unpack_proto_method_response, ProtoMsg
from .grpc_interface import grpc_register_function, grpc_register_method, grpc_function_call, grpc_method_call, remote_connected
from .instance_ref import init_remoteclass, assign_ref_to_instance, del_instance

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

def function_decorator(func: Callable[P, R]) -> Callable[P, R]:
    def function_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
        args = unpack_proto_function_request(func, proto_req)
        result = func(**args) # type: ignore[call-arg]
        return pack_proto_function_response(func, result)

    grpc_register_function(func, function_handler)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
        if remote_connected():
            proto_req = pack_proto_function_request(func, *args, **kwargs)
            proto_res = grpc_function_call(func, proto_req)
            return unpack_proto_function_response(func, proto_res)
        else:
            return func(*args, **kwargs)

    compile_function(func)

    return cast(Callable[P, R], wrapper)

def dataclass_decorator(cls: Type[T]) -> Type[T]:
    datacls = dataclasses.dataclass(cls)
    compile_dataclass(datacls)
    return datacls

class method_decorator(Generic[P, R]):
    def __init__(self, method: Callable[P, R]):
        self.method = method
        functools.update_wrapper(self, method)

    def __set_name__(self, cls: Type[T], method_name: str) -> None:
        self.cls = cls
        self.method_name = method_name

        method = self.method

        def method_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
            args = unpack_proto_method_request(cls, method, proto_req)
            result = method(**args) # type: ignore[call-arg]
            return pack_proto_method_response(cls, method, result)

        grpc_register_method(cls, method, method_handler)

        compile_remoteclass_method(cls, method)

    def __get__(self, instance: T, cls: Type[T]) -> Callable[..., R]:
        return MethodType(self, instance)

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        if remote_connected():
            proto_req = pack_proto_method_request(self.cls, self.method, *args, **kwargs)
            proto_res = grpc_method_call(self.cls, self.method, proto_req)
            return unpack_proto_method_response(self.cls, self.method, proto_res) # type: ignore[no-any-return]
        else:
            return self.method(*args, **kwargs)

def remoteclass_decorator(cls: Type[T]) -> Type[T]:
    init_remoteclass(cls)

    init_method = getattr(cls, "__init__")

    def init_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
        args = unpack_proto_method_request(cls, init_method, proto_req)
        obj = cls(**args)
        return pack_proto_method_response(cls, init_method, obj)

    grpc_register_method(cls, init_method, init_handler)

    @functools.wraps(init_method)
    def init_wrapper(*args: Any, **kwargs: Any) -> None:
        if remote_connected():
            proto_req = pack_proto_method_request(cls, init_method, *args, **kwargs)
            proto_res = grpc_method_call(cls, init_method, proto_req)
            instance_ref = unpack_proto_method_response(cls, init_method, proto_res)
            assign_ref_to_instance(cls, args[0], instance_ref, False)
        else:
            init_method(*args, **kwargs)

    setattr(cls, "__init__", init_wrapper)

    compile_remoteclass_init(cls)

    del_method = getattr(cls, "__del__")

    def del_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
        args = unpack_proto_method_request(cls, del_method, proto_req)
        del_instance(cls, list(args.values())[0])
        return pack_proto_method_response(cls, del_method, None)

    grpc_register_method(cls, del_method, del_handler)

    @functools.wraps(del_method)
    def del_wrapper(self: Any) -> None:
        if remote_connected():
            proto_req = pack_proto_method_request(cls, del_method, self)
            grpc_method_call(cls, del_method, proto_req)
        else:
            del_method(self)

    setattr(cls, "__del__", del_wrapper)

    compile_remoteclass_del(cls)

    compile_remoteclass(cls)

    return cls
