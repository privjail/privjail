from typing import Any, TypeGuard
import numpy as _np

class DPError(Exception):
    pass

integer = int | _np.integer[Any]
floating = float | _np.floating[Any]
realnum = integer | floating

def is_integer(x: Any) -> TypeGuard[integer]:
    return isinstance(x, (int, _np.integer))

def is_floating(x: Any) -> TypeGuard[floating]:
    return isinstance(x, (float, _np.floating))

def is_realnum(x: Any) -> TypeGuard[realnum]:
    return is_integer(x) or is_floating(x)
