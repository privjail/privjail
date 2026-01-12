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

from types import EllipsisType
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

        egrpc.disconnect()

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

@egrpc.function
def func9(x: Tuple[int, ...]) -> Tuple[float, ...]:
    return tuple(float(n) for n in x)

@egrpc.function
def func_slice(s: slice) -> slice:
    return s

@egrpc.function
def func_ellipsis(e: EllipsisType) -> EllipsisType:
    return e

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
    assert func8() is None
    assert func9((1, 2, 3)) == (1.0, 2.0, 3.0)
    assert func_slice(slice(1, 10, 2)) == slice(1, 10, 2)
    assert func_slice(slice(None, 5)) == slice(None, 5)
    assert func_slice(slice(3, None)) == slice(3, None)
    assert func_slice(slice(None)) == slice(None)
    assert func_ellipsis(...) is ...

@egrpc.multifunction
def mfunc1(x: int | float, y: int | float) -> float:
    return x + y

@mfunc1.register
def _(x: int, y: int) -> int:
    return x + y

def test_multifunction(server: Any) -> None:
    assert mfunc1(1, 2) == 3
    assert isinstance(mfunc1(1, 2), int)
    assert mfunc1(1.1, 2) == pytest.approx(3.1)
    assert isinstance(mfunc1(1.1, 2), float)
    assert mfunc1(1.1, 2.2) == pytest.approx(3.3)
    assert isinstance(mfunc1(1.1, 2.2), float)

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
    d = Data(3, 2.2)
    assert dcfunc1(d) == pytest.approx(3 * 2.2)
    assert dcfunc2(d, d) == True
    assert dcfunc2(d, Data(2, 2.2)) == False

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

    @egrpc.multimethod
    def __mul__(self, other: int | float) -> "Point":
        return Point(self.x * other, self.y * other, self.z * other)

    @__mul__.register
    def _(self, other: "Point") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

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
    assert p1 * p2 == pytest.approx(15.4)

    with pytest.raises(AttributeError):
        p1.x

@egrpc.remoteclass
class Hoge():
    def __init__(self, x: int):
        self.x = x

    @egrpc.property
    def value(self) -> int:
        return self.x

    @value.setter
    def value(self, x: int) -> None:
        self.x = x

    @egrpc.method
    def fuga(self) -> "Fuga":
        return Fuga(str(self.x))

@egrpc.remoteclass
class Fuga():
    def __init__(self, x: str):
        self.x = x

    @egrpc.property
    def value(self) -> str:
        return self.x

    @value.setter
    def value(self, x: str) -> None:
        self.x = x

    @egrpc.method
    def hoge(self) -> "Hoge":
        return Hoge(int(self.x))

@egrpc.function
def gen_hoge(x: int) -> Hoge:
    return Hoge(x)

def test_remoteclass2(server: Any) -> None:
    hoge = gen_hoge(1)
    fuga = hoge.fuga()
    hoge_ = fuga.hoge()
    assert hoge.value == hoge_.value

    hoge.value = 2
    assert hoge.value == 2
    assert hoge.fuga().value == "2"

@egrpc.remoteclass
class Animal:
    def __init__(self, name: str):
        self._name = name

    @egrpc.property
    def name(self) -> str:
        return self._name

    @egrpc.method
    def speak(self) -> str:
        return "..."

@egrpc.remoteclass
class Dog(Animal):
    def __init__(self, name: str):
        super().__init__(name)

    @egrpc.method
    def speak(self) -> str:
        return "woof"

@egrpc.remoteclass
class Cat(Animal):
    def __init__(self, name: str):
        super().__init__(name)

    @egrpc.method
    def speak(self) -> str:
        return "meow"

@egrpc.remoteclass
class Mammal(Animal):
    def __init__(self, name: str, legs: int):
        super().__init__(name)
        self._legs = legs

    @egrpc.property
    def legs(self) -> int:
        return self._legs

@egrpc.remoteclass
class Horse(Mammal):
    def __init__(self, name: str):
        super().__init__(name, 4)

    @egrpc.method
    def speak(self) -> str:
        return "neigh"

@egrpc.function
def create_animal(kind: str, name: str) -> Animal:
    if kind == "dog":
        return Dog(name)
    elif kind == "cat":
        return Cat(name)
    elif kind == "horse":
        return Horse(name)
    else:
        return Animal(name)

@egrpc.function
def get_animal_name(animal: Animal) -> str:
    return animal.name

def test_inheritance(server: Any) -> None:
    dog = create_animal("dog", "Pochi")
    assert isinstance(dog, Dog)
    assert dog.name == "Pochi"
    assert dog.speak() == "woof"
    assert get_animal_name(dog) == "Pochi"

    cat = create_animal("cat", "Tama")
    assert isinstance(cat, Cat)
    assert cat.name == "Tama"
    assert cat.speak() == "meow"
    assert get_animal_name(cat) == "Tama"

    horse = create_animal("horse", "Uma")
    assert isinstance(horse, Horse)
    assert horse.name == "Uma"
    assert horse.speak() == "neigh"
    assert horse.legs == 4
    assert get_animal_name(horse) == "Uma"

    animal = create_animal("unknown", "X")
    assert isinstance(animal, Animal)
    assert animal.name == "X"
    assert animal.speak() == "..."
