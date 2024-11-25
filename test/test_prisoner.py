import pytest
import uuid
import pripri

def new_sensitive_int(value: int) -> pripri.SensitiveInt:
    return pripri.SensitiveInt(value, distance=pripri.Distance(1), root_name=str(uuid.uuid4()))

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
