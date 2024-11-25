import pytest
import uuid
import pripri

def new_sensitive_int(value: int) -> pripri.SensitiveInt:
    return pripri.SensitiveInt(value, distance=pripri.Distance(1), root_name=str(uuid.uuid4())) + 0

def test_sensitive_real_number() -> None:
    x = new_sensitive_int(12)
    assert isinstance(x, pripri.SensitiveInt)

    y = x * (1 / 12)
    assert isinstance(y, pripri.SensitiveFloat)

    x = x + 1
    assert x._value == 13
    assert x.distance.max() == 1

    x = 1 + x
    assert x._value == 14
    assert x.distance.max() == 1

    x += 1
    assert x._value == 15
    assert x.distance.max() == 1

    x *= 3
    assert x._value == 45
    assert x.distance.max() == 3

    x = x * 2 + 1
    assert x._value == 91
    assert x.distance.max() == 6

    x -= 90
    assert x._value == 1
    assert x.distance.max() == 6

    with pytest.raises(pripri.DPError):
        x * x # type: ignore

    z = x + y
    assert z._value == pytest.approx(2.0)
    assert z.distance.max() == pytest.approx(6.0 + 1 / 12)

    z = x - 2 * y
    assert z._value == pytest.approx(-1.0)
    assert z.distance.max() == pytest.approx(6.0 + 1 / 6)

    with pytest.raises(pripri.DPError):
        x * y # type: ignore

def test_min_max() -> None:
    x = new_sensitive_int(11)
    y = x - 1
    z = x + 2

    assert pripri.max(x, y)._value == 11
    assert pripri.max(x, y).distance.max() == 1

    assert pripri.min(x, y)._value == 10
    assert pripri.min(x, y).distance.max() == 1

    assert pripri.max(x, y, z)._value == 13
    assert pripri.max(x, y, z).distance.max() == 1

    assert pripri.min(x, y, z)._value == 10
    assert pripri.min(x, y, z).distance.max() == 1

    z *= 2

    assert pripri.max(x, y, z)._value == 26
    assert pripri.max(x, y, z).distance.max() == 2

    assert pripri.min(x, y, z)._value == 10
    assert pripri.min(x, y, z).distance.max() == 2

    assert pripri.max([x, y, z])._value == 26
    assert pripri.max([x, y, z]).distance.max() == 2

    assert pripri.min([x, y, z])._value == 10
    assert pripri.min([x, y, z]).distance.max() == 2

    assert type(pripri.max(x * 3.0, y, z)) == pripri.SensitiveFloat
    assert pripri.max(x * 3.0, y, z)._value == pytest.approx(33.0)
    assert pripri.max(x * 3.0, y, z).distance.max() == pytest.approx(3.0)

    assert type(pripri.min(x * 3.0, y, z)) == pripri.SensitiveFloat
    assert pripri.min(x * 3.0, y, z)._value == pytest.approx(10.0)
    assert pripri.min(x * 3.0, y, z).distance.max() == pytest.approx(3.0)

    assert type(pripri.max([x, y, z - 25.0])) == pripri.SensitiveFloat
    assert pripri.max([x, y, z - 25.0])._value == pytest.approx(11.0)
    assert pripri.max([x, y, z - 25.0]).distance.max() == pytest.approx(2.0)

    assert type(pripri.min([x, y, z - 25.0])) == pripri.SensitiveFloat
    assert pripri.min([x, y, z - 25.0])._value == pytest.approx(1.0)
    assert pripri.min([x, y, z - 25.0]).distance.max() == pytest.approx(2.0)

    with pytest.raises(TypeError): pripri.max()
    with pytest.raises(TypeError): pripri.min()

    with pytest.raises(TypeError): pripri.max(x)
    with pytest.raises(TypeError): pripri.min(x)

    with pytest.raises(ValueError): pripri.max([])
    with pytest.raises(ValueError): pripri.min([])
