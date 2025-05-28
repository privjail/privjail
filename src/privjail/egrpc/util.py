# Copyright 2025 TOYOTA MOTOR CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import get_type_hints, get_origin, get_args, Union, List, Tuple, Dict, Any, TypeVar, Callable, Type, ParamSpec
from types import UnionType
from collections.abc import Sequence, Mapping
import inspect

# TODO: make egrpc independent of numpy
import numpy as _np

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")

TypeHint = Any

# https://github.com/python/mypy/issues/15630
def my_get_origin(type_hint: Any) -> Any:
    return get_origin(type_hint)

def is_subtype(type_hint1: TypeHint, type_hint2: TypeHint) -> bool:
    type_origin1 = my_get_origin(type_hint1)
    type_origin2 = my_get_origin(type_hint2)
    type_args1 = get_args(type_hint1)
    type_args2 = get_args(type_hint2)

    if type_origin1 is None and type_origin2 is None:
        try:
            return issubclass(type_hint1, type_hint2)
        except TypeError:
            return type_hint1 is type_hint2

    elif type_origin1 in (Union, UnionType) and type_origin2 in (Union, UnionType):
        return all(any(is_subtype(th1, th2) for th2 in type_args2) for th1 in type_args1)

    elif type_origin2 in (Union, UnionType):
        return any(is_subtype(type_hint1, th2) for th2 in type_args2)

    elif type_origin1 in (Union, UnionType):
        return all(is_subtype(th1, type_hint2) for th1 in type_args1)

    elif type_origin1 is _np.integer and type_origin2 is None:
        return type_hint2 in (int, float)

    elif type_origin1 is _np.floating and type_origin2 is None:
        return type_hint2 is float

    elif type_origin1 is None and type_origin2 in (_np.integer, _np.floating):
        return False

    elif type_origin1 == type_origin2:
        # TODO: seriously consider covariant, contravariant, Sequence, Mapping, ...
        return (
            len(type_args1) != len(type_args2)
            and all(is_subtype(th1, th2) for th1, th2 in zip(type_args1, type_args2))
        )

    else:
        return False

def is_type_match(obj: Any, type_hint: TypeHint) -> bool:
    type_origin = my_get_origin(type_hint)
    type_args = get_args(type_hint)

    if type_origin is None:
        return isinstance(obj, type_hint)

    elif type_origin in (Union, UnionType):
        return any(is_type_match(obj, th) for th in type_args)

    elif type_origin in (tuple, Tuple):
        return (
            isinstance(obj, tuple)
            and len(obj) == len(type_args)
            and all(is_type_match(o, th) for o, th in zip(obj, type_args))
        )

    elif type_origin in (list, List, Sequence):
        return (
            isinstance(obj, list)
            and all(is_type_match(o, type_args[0]) for o in obj)
        )

    elif type_origin in (dict, Dict, Mapping):
        return (
            isinstance(obj, dict)
            and all(is_type_match(k, type_args[0]) for k in obj.keys())
            and all(is_type_match(v, type_args[1]) for v in obj.values())
        )

    elif type_origin in (_np.integer, _np.floating):
        # TODO: consider type args (T in np.integer[T])
        return isinstance(obj, (_np.integer, _np.floating))

    else:
        return (
            hasattr(obj, "__orig_class__")
            and all(is_subtype(th1, th2) for th1, th2 in zip(get_args(obj.__orig_class__), type_args))
        )

def get_function_typed_params(func: Callable[P, R]) -> dict[str, TypeHint]:
    type_hints = get_type_hints(func)
    param_names = list(inspect.signature(func).parameters.keys())
    return {param_name: type_hints[param_name] for param_name in param_names}

def get_function_return_type(func: Callable[P, R]) -> TypeHint:
    type_hints = get_type_hints(func)
    return type_hints["return"]

def get_class_typed_members(cls: Type[T]) -> dict[str, TypeHint]:
    return get_type_hints(cls)

def get_method_typed_params(cls: Type[T], method: Callable[P, R]) -> dict[str, TypeHint]:
    type_hints = get_type_hints(method)
    param_names = list(inspect.signature(method).parameters.keys())
    return {param_names[0]: cls,
            **{param_name: type_hints[param_name] for param_name in param_names[1:]}}

def get_method_return_type(cls: Type[T], method: Callable[P, R]) -> TypeHint:
    type_hints = get_type_hints(method)
    return type_hints["return"]

def normalize_args(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments
