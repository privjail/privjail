from typing import get_type_hints, get_origin, get_args, Union, List, Tuple, Dict
import types
import os
import inspect

egrpc_mode = os.getenv("EGRPC_MODE", "client")

def is_type_match(obj, type_hint) -> bool:
    type_origin = get_origin(type_hint)
    type_args = get_args(type_hint)

    if type_origin is None:
        return isinstance(obj, type_hint)

    elif type_origin in (Union, types.UnionType):
        return any(is_type_match(obj, th) for th in type_args)

    elif type_origin in (tuple, Tuple):
        return (
            isinstance(obj, tuple)
            and len(obj) == len(type_args)
            and all(is_type_match(o, th) for o, th in zip(obj, type_args))
        )

    elif type_origin in (list, List):
        return (
            isinstance(obj, list)
            and all(is_type_match(o, type_args[0]) for o in obj)
        )

    elif type_origin in (dict, Dict):
        return (
            isinstance(obj, dict)
            and all(is_type_match(k, type_args[0]) for k in obj.keys())
            and all(is_type_match(v, type_args[1]) for v in obj.values())
        )

    else:
        raise Exception

def get_function_typed_params(func):
    type_hints = get_type_hints(func)
    param_names = list(inspect.signature(func).parameters.keys())
    return {param_name: type_hints[param_name] for param_name in param_names}

def get_function_return_type(func):
    type_hints = get_type_hints(func)
    return type_hints["return"]

def get_class_typed_members(cls):
    return get_type_hints(cls)

def get_method_self_name(cls, method):
    return list(inspect.signature(method).parameters.keys())[0]

def get_method_typed_params(cls, method):
    globalns = {cls.__name__: cls, **globals()}
    type_hints = get_type_hints(method, globalns=globalns)
    param_names = list(inspect.signature(method).parameters.keys())
    return {param_name: type_hints[param_name] for param_name in param_names[1:]}

def get_method_return_type(cls, method):
    globalns = {cls.__name__: cls, **globals()}
    type_hints = get_type_hints(method, globalns=globalns)
    return type_hints["return"]
