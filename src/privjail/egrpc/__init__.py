from .entry_point import serve, connect
from .decorator import function_decorator as function, dataclass_decorator as dataclass, remoteclass_decorator as remoteclass, method_decorator as method

__all__ = [
    "serve",
    "connect",
    "function",
    "dataclass",
    "remoteclass",
    "method",
]
