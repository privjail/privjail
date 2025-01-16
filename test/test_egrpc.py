from typing import Union, List, Tuple, Dict, Optional, Iterator, Any
import sys
import os
import multiprocessing
import time
import math
import pytest
from privjail import egrpc

env_name = "client"

def serve(port: int, error_queue: Any) -> None:
    global env_name
    env_name = "server"

    # redirect stdout to parent process
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

    try:
        egrpc.serve(port)
    except AssertionError as e:
        error_queue.put(str(e))
    except Exception as e:
        error_queue.put(f"Unexpected error: {e}")

@pytest.fixture(scope="module")
def server() -> Iterator[None]:
    port = 12345

    error_queue: Any = multiprocessing.Queue()

    server_process = multiprocessing.Process(target=serve, args=(port, error_queue))
    server_process.start()

    time.sleep(1)

    if not server_process.is_alive():
        raise RuntimeError("Server failed to start.")

    egrpc.connect("localhost", port)

    try:
        yield

    finally:
        while not error_queue.empty():
            error_message = error_queue.get()
            pytest.fail(f"Server process error: {error_message}")

        server_process.terminate()
        server_process.join()

@egrpc.function
def get_gvar() -> str:
    return env_name

def test_remote_exec(server: Any) -> None:
    assert env_name == "client"
    assert get_gvar() == "server"

@egrpc.function
def func1(name: str, age: int) -> str:
    return f"{name}: {age}"

@egrpc.function
def func2(name: str, weight: int | float) -> str:
    return f"{name}: {weight:.2f}"

@egrpc.function
def func3(x: Union[int, float]) -> int | float:
    return x * x

@egrpc.function
def func4(lst: list[int] | list[str], n: int) -> list[int] | List[str]:
    return lst * n

@egrpc.function
def func5(lst: List[int | str], n: int) -> list[int | str]:
    return lst * n

@egrpc.function
def func6(tup: tuple[int, str]) -> Tuple[str, int]:
    return tup[1], tup[0]

@egrpc.function
def func7(d: Dict[str, int]) -> dict[str, list[str]]:
    return {k: [k] * v for k, v in d.items()}

@egrpc.function
def func8(x: Optional[int] = None) -> int | None:
    return x * x if x is not None else None

def test_function(server: Any) -> None:
    assert func1("Alice", 30) == "Alice: 30"
    assert func2("Bob", 60) == "Bob: 60.00"
    assert func2("Bob", 60.2) == "Bob: 60.20"
    assert func3(2) == 4
    assert func3(4.3) == pytest.approx(4.3 * 4.3)
    assert func4([1, 2, 3], 2) == [1, 2, 3, 1, 2, 3]
    assert func4(["a", "b", "c"], 2) == ["a", "b", "c", "a", "b", "c"]
    assert func5([1, "b", 3], 2) == [1, "b", 3, 1, "b", 3]
    assert func6((1, "a")) == ("a", 1)
    assert func7({"a": 1, "b": 2}) == {"a": ["a"], "b": ["b", "b"]}
    assert func8(2) == 4
    assert func8() == None

@egrpc.dataclass
class Data():
    x: int
    y: float

@egrpc.function
def dcfunc1(d: Data) -> float:
    return d.x * d.y

@egrpc.function
def dcfunc2(d1: Data, d2: Data) -> bool:
    return d1 == d2

def test_dataclass(server: Any) -> None:
    d = Data(3, 2.2) # type: ignore[call-arg]
    assert dcfunc1(d) == pytest.approx(3 * 2.2)
    assert dcfunc2(d, d) == True
    assert dcfunc2(d, Data(2, 2.2)) == False # type: ignore[call-arg]

@egrpc.remoteclass
class Point():
    @egrpc.method
    def __init__(self, x: int | float, y: int | float, z: int | float):
        self.x = x
        self.y = y
        self.z = z

    @egrpc.method
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @egrpc.method
    def __str__(self) -> str:
        return ",".join([str(self.x), str(self.y), str(self.z)])

    @egrpc.method
    def __mul__(self, other: int | float) -> "Point":
        return Point(self.x * other, self.y * other, self.z * other)

@egrpc.function
def identity(p: Point) -> Point:
    return p

@egrpc.function
def norm(p: Point) -> float:
    return p.norm()

def test_remoteclass(server: Any) -> None:
    p1 = Point(1, 2, 3)
    p2 = Point(1.1, 2.2, 3.3)

    assert p1 != p2
    assert identity(p1) == p1
    assert p1.norm() == pytest.approx(math.sqrt(14))
    assert norm(p1) == pytest.approx(math.sqrt(14))
    assert str(p1) == "1,2,3"
    assert (p1 * 2) != p1
    assert (p1 * 2).norm() == pytest.approx(math.sqrt(56))

    with pytest.raises(AttributeError):
        p1.x
