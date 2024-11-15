import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
distance = importlib.import_module("pripri.distance")

def test_distance() -> None:
    d = distance.Distance(1)
    assert d.max() == 1

    d = distance.Distance(d.expr * 2)
    assert d.max() == 2

    d = distance.Distance(d.expr + 1)
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

    d = distance.Distance(d.expr + y, d.constraints)
    assert d.max() == 1

    d = distance.Distance(d.expr + z, d.constraints)
    assert d.max() == 1

    d = distance.Distance(d.expr * 2, d.constraints)
    assert d.max() == 2

    d = distance.Distance(d.expr + 1, d.constraints)
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

    d = distance.Distance(d.expr + y_ + z_, d.constraints)
    assert d.max() == 1

    d = distance.Distance(d.expr * 4, d.constraints)
    assert d.max() == 4

    d = distance.Distance(d.expr + x, d.constraints)
    assert d.max() == 5

    d = distance.Distance(d.expr + y + z, d.constraints)
    assert d.max() == 5

    d = distance.Distance(d.expr * 2, d.constraints)
    assert d.max() == 10

    d = distance.Distance(d.expr + x_, d.constraints)
    assert d.max() == 11
