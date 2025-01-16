from typing import get_type_hints, get_origin, get_args, Union, List, Tuple, Dict, Any, TypeVar, Callable, Type
import types
import os
import inspect

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

TypeHint = Any

egrpc_mode = os.getenv("EGRPC_MODE", "client")

# https://github.com/python/mypy/issues/15630
def my_get_origin(type_hint: Any) -> Any:
    return get_origin(type_hint)

def is_type_match(obj: Any, type_hint: TypeHint) -> bool:
    type_origin = my_get_origin(type_hint)
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

def get_function_typed_params(func: F) -> dict[str, TypeHint]:
    type_hints = get_type_hints(func)
    param_names = list(inspect.signature(func).parameters.keys())
    return {param_name: type_hints[param_name] for param_name in param_names}

def get_function_return_type(func: F) -> TypeHint:
    type_hints = get_type_hints(func)
    return type_hints["return"]

def get_class_typed_members(cls: Type[T]) -> dict[str, TypeHint]:
    return get_type_hints(cls)

def get_method_typed_params(cls: Type[T], method: F) -> dict[str, TypeHint]:
    globalns = {cls.__name__: cls, **globals()}
    type_hints = get_type_hints(method, globalns=globalns)
    param_names = list(inspect.signature(method).parameters.keys())
    return {param_names[0]: cls,
            **{param_name: type_hints[param_name] for param_name in param_names[1:]}}

def get_method_return_type(cls: Type[T], method: F) -> TypeHint:
    globalns = {cls.__name__: cls, **globals()}
    type_hints = get_type_hints(method, globalns=globalns)
    return type_hints["return"]

def normalize_args(func: F, *args: Any, **kwargs: Any) -> dict[str, Any]:
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments
