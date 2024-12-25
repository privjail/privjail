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

    constraints = {distance.Constraint(frozenset({x, y, z}), 1)}

    dx = distance.Distance(x, constraints)
    dy = distance.Distance(y, constraints)
    dz = distance.Distance(z, constraints)

    assert dx.max() == 1

    d = dx + dy
    assert d.max() == 1

    d = d + dz
    assert d.max() == 1

    d = d * 2
    assert d.max() == 2

    d = d + 1
    assert d.max() == 3

    x_ = distance.new_distance_var()
    y_ = distance.new_distance_var()
    z_ = distance.new_distance_var()

    constraints |= {distance.Constraint(frozenset({x_, y_, z_}), x)}

    dx_ = distance.Distance(x_, constraints)
    dy_ = distance.Distance(y_, constraints)
    dz_ = distance.Distance(z_, constraints)

    assert dx_.max() == 1

    d = dx_ + dy_ + dz_
    assert d.max() == 1

    d = d * 4
    assert d.max() == 4

    d = d + dx
    assert d.max() == 5

    d = d + dy + dz
    assert d.max() == 5

    d = d * 2
    assert d.max() == 10

    d = d + dx_
    assert d.max() == 11
