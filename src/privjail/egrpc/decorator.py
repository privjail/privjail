from typing import TypeVar, Callable, Type, Any, cast, ParamSpec
import functools
import dataclasses

import multimethod

from .compiler import compile_function, compile_dataclass, compile_remoteclass, compile_remoteclass_init, compile_remoteclass_method, compile_remoteclass_del
from .proto_interface import pack_proto_function_request, pack_proto_function_response, unpack_proto_function_request, unpack_proto_function_response, pack_proto_method_request, pack_proto_method_response, unpack_proto_method_request, unpack_proto_method_response, ProtoMsg
from .grpc_interface import grpc_register_function, grpc_register_method, grpc_function_call, grpc_method_call, remote_connected
from .instance_ref import init_remoteclass, assign_ref_to_instance, del_instance

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

def function_decorator(func: Callable[P, R]) -> Callable[P, R]:
    compile_function(func)

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

    return cast(Callable[P, R], wrapper)

def dataclass_decorator(cls: Type[T]) -> Type[T]:
    datacls = dataclasses.dataclass(cls)
    compile_dataclass(datacls)
    return datacls

def method_decorator(method: Callable[P, R]) -> Callable[P, R]:
    setattr(method, "__egrpc_enabled", True)
    return method

class multimethod_decorator:
    methods: list[Callable[..., Any]]

    def __init__(self, method: Callable[P, R]):
        setattr(self, "__egrpc_enabled", True)
        self.methods = [method]

    def register(self, method: Callable[P, R]) -> None:
        self.methods.append(method)
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("@egrpc.remoteclass is missing")

def remoteclass_decorator(cls: Type[T]) -> Type[T]:
    init_remoteclass(cls)
    methods = {k: v for k, v in cls.__dict__.items() if hasattr(v, "__egrpc_enabled")}

    for method_name, method in methods.items():
        if isinstance(method, multimethod_decorator):
            register_remoteclass_multimethod(cls, method, method_name)
        else:
            if method_name == "__init__":
                register_remoteclass_init(cls)
            else:
                register_remoteclass_method(cls, method, method_name)

    register_remoteclass_del(cls)

    compile_remoteclass(cls)

    return cls

def register_remoteclass_init(cls: Type[T]) -> None:
    init_method = getattr(cls, "__init__")

    compile_remoteclass_init(cls, init_method)

    def init_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
        args = unpack_proto_method_request(cls, init_method, proto_req)
        obj = cls(**args)
        return pack_proto_method_response(cls, init_method, obj)

    grpc_register_method(cls, init_method, init_handler)

    @functools.wraps(init_method)
    def init_wrapper(*args: Any, init_method: Callable[P, R] = init_method, **kwargs: Any) -> None:
        if remote_connected():
            proto_req = pack_proto_method_request(cls, init_method, *args, **kwargs)
            proto_res = grpc_method_call(cls, init_method, proto_req)
            instance_ref = unpack_proto_method_response(cls, init_method, proto_res)
            assign_ref_to_instance(cls, args[0], instance_ref, False)
        else:
            init_method(*args, **kwargs)

    setattr(cls, "__init__", init_wrapper)

def register_remoteclass_del(cls: Type[T]) -> None:
    del_method = getattr(cls, "__del__")

    compile_remoteclass_del(cls, del_method)

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

def register_remoteclass_method(cls: Type[T], method: Callable[P, R], method_name: str) -> None:
    compile_remoteclass_method(cls, method)

    def method_handler(self: Any, proto_req: ProtoMsg, context: Any) -> ProtoMsg:
        args = unpack_proto_method_request(cls, method, proto_req)
        result = method(**args) # type: ignore[call-arg]
        return pack_proto_method_response(cls, method, result)

    grpc_register_method(cls, method, method_handler)

    @functools.wraps(method)
    def method_wrapper(*args: Any, **kwargs: Any) -> Any:
        if remote_connected():
            proto_req = pack_proto_method_request(cls, method, *args, **kwargs)
            proto_res = grpc_method_call(cls, method, proto_req)
            return unpack_proto_method_response(cls, method, proto_res)
        else:
            return method(*args, **kwargs)

    setattr(cls, method_name, method_wrapper)

def register_remoteclass_multimethod(cls: Type[T], mmd: multimethod_decorator, method_name: str) -> None:
    qualname = mmd.methods[0].__qualname__

    for i, method in enumerate(mmd.methods):
        method.__qualname__ = f"{qualname}.{i}"

        compile_remoteclass_method(cls, method)

        def method_handler(self: Any, proto_req: ProtoMsg, context: Any, method: Callable[P, R] = method) -> ProtoMsg:
            args = unpack_proto_method_request(cls, method, proto_req)
            result = method(**args) # type: ignore[call-arg]
            return pack_proto_method_response(cls, method, result)

        grpc_register_method(cls, method, method_handler)

    def gen_wrapper(method: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(method)
        def method_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if remote_connected():
                proto_req = pack_proto_method_request(cls, method, *args, **kwargs)
                proto_res = grpc_method_call(cls, method, proto_req)
                return unpack_proto_method_response(cls, method, proto_res) # type: ignore[no-any-return]
            else:
                return method(*args, **kwargs)

        return method_wrapper

    mm = multimethod.multimethod(gen_wrapper(mmd.methods[0]))

    for method in mmd.methods[1:]:
        mm.register(gen_wrapper(method))

    setattr(cls, method_name, mm)
