from typing import TypeVar, Generic

T = TypeVar("T")

class Prisoner(Generic[T]):
    def __init__(self, value: T, sensitivity: float):
        self._value = value
        self.sensitivity = sensitivity

    def __str__(self) -> str:
        return f"Prisoner({type(self._value)}, sensitivity={self.sensitivity})"

    def __repr__(self) -> str:
        return f"Prisoner({type(self._value)}, sensitivity={self.sensitivity})"
