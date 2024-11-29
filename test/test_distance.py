import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
distance = importlib.import_module("privjail.distance")

def test_distance() -> None:
    d = distance.Distance(1)
    assert d.max() == 1

    d = d * 2
    assert d.max() == 2

    d = d + 1
    assert d.max() == 3

    x = distance.new_distance_var()
    y = distance.new_distance_var()
    z = distance.new_distance_var()

    constraints = [
        0 <= x,
        0 <= y,
        0 <= z,
        x + y + z <= 1,
    ]

    d = distance.Distance(x, constraints)
    assert d.max() == 1

    d = d + y
    assert d.max() == 1

    d = d + z
    assert d.max() == 1

    d = d * 2
    assert d.max() == 2

    d = d + 1
    assert d.max() == 3

    x_ = distance.new_distance_var()
    y_ = distance.new_distance_var()
    z_ = distance.new_distance_var()

    constraints += [
        0 <= x_,
        0 <= y_,
        0 <= z_,
        x_ + y_ + z_ <= x,
    ]

    d = distance.Distance(x_, constraints)
    assert d.max() == 1

    d = d + y_ + z_
    assert d.max() == 1

    d = d * 4
    assert d.max() == 4

    d = d + x
    assert d.max() == 5

    d = d + y + z
    assert d.max() == 5

    d = d * 2
    assert d.max() == 10

    d = d + x_
    assert d.max() == 11
