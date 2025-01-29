from typing import TypeVar, Callable, Type, Any

T = TypeVar("T")

def proto_base_name(func: Callable[..., Any]) -> str:
    return "".join([x.capitalize() for x in func.__qualname__.split(".")])

def proto_function_service_name(func: Callable[..., Any]) -> str:
    return f"{proto_base_name(func)}FunctionService"

def proto_function_rpc_name(func: Callable[..., Any]) -> str:
    return proto_base_name(func)

def proto_function_req_name(func: Callable[..., Any]) -> str:
    return f"{proto_base_name(func)}FunctionRequest"

def proto_function_res_name(func: Callable[..., Any]) -> str:
    return f"{proto_base_name(func)}FunctionResponse"

def proto_dataclass_name(cls: Type[T]) -> str:
    return f"{proto_base_name(cls)}DataClassMessage"

def proto_remoteclass_service_name(cls: Type[T]) -> str:
    return f"{proto_base_name(cls)}RemoteClassService"

def proto_method_rpc_name(cls: Type[T], method: Callable[..., Any]) -> str:
    if method.__name__ == "__init__":
        return f"{proto_base_name(cls)}Init"
    elif method.__name__ == "__del__":
        return f"{proto_base_name(cls)}Del"
    else:
        return f"{proto_base_name(method)}Method"

def proto_method_req_name(cls: Type[T], method: Callable[..., Any]) -> str:
    if method.__name__ == "__init__":
        return f"{proto_base_name(cls)}InitRemoteClassRequest"
    elif method.__name__ == "__del__":
        return f"{proto_base_name(cls)}DelRemoteClassRequest"
    else:
        return f"{proto_base_name(method)}MethodRemoteClassRequest"

def proto_method_res_name(cls: Type[T], method: Callable[..., Any]) -> str:
    if method.__name__ == "__init__":
        return f"{proto_base_name(cls)}InitRemoteClassResponse"
    elif method.__name__ == "__del__":
        return f"{proto_base_name(cls)}DelRemoteClassResponse"
    else:
        return f"{proto_base_name(method)}MethodRemoteClassResponse"

def proto_instance_ref_name(cls: Type[T]) -> str:
    return f"{proto_base_name(cls)}InstanceMessage"
