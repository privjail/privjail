from typing import Any
import pytest
import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
provenance = importlib.import_module("privjail.provenance")

@pytest.fixture(autouse=True)
def setup() -> Any:
    provenance.clear_global_states()
    yield

def test_provenance_accumulation() -> None:
    pe0 = provenance.new_provenance_root("foo")
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 0

    provenance.consume_privacy_budget([pe0], 10)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 10

    pe1 = provenance.new_provenance_node([pe0], "inclusive")
    pe2 = provenance.new_provenance_node([pe0], "inclusive")

    provenance.consume_privacy_budget([pe1], 20)
    provenance.consume_privacy_budget([pe2], 30)

    assert pe1.consumed_privacy_budget == 20
    assert pe2.consumed_privacy_budget == 30
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 60

    pe1e = provenance.new_provenance_node([pe1], "exclusive")
    pe1e1 = provenance.new_provenance_node([pe1e], "inclusive")
    pe1e2 = provenance.new_provenance_node([pe1e], "inclusive")

    provenance.consume_privacy_budget([pe1e1], 50)
    provenance.consume_privacy_budget([pe1e2], 30)

    assert pe1e1.consumed_privacy_budget == 50
    assert pe1e2.consumed_privacy_budget == 30
    assert pe1e.consumed_privacy_budget == 50
    assert pe1.consumed_privacy_budget == 70
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 110

    provenance.consume_privacy_budget([pe1e2], 30)

    assert pe1e1.consumed_privacy_budget == 50
    assert pe1e2.consumed_privacy_budget == 60
    assert pe1e.consumed_privacy_budget == 60
    assert pe1.consumed_privacy_budget == 80
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 120

    pe2e = provenance.new_provenance_node([pe2], "exclusive")
    pe2e1 = provenance.new_provenance_node([pe2e], "inclusive")
    pe2e2 = provenance.new_provenance_node([pe2e], "inclusive")
    pe2e3 = provenance.new_provenance_node([pe2e], "inclusive")
    pe2e1_2e2_1 = provenance.new_provenance_node([pe2e1, pe2e2], "inclusive")
    pe2e1_2e3_1 = provenance.new_provenance_node([pe2e1, pe2e3], "inclusive")
    pe2e2_2e3_1 = provenance.new_provenance_node([pe2e2, pe2e3], "inclusive")

    provenance.consume_privacy_budget([pe2e1_2e2_1], 10)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 130
    provenance.consume_privacy_budget([pe2e2_2e3_1], 10)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 140
    provenance.consume_privacy_budget([pe2e1_2e3_1], 10)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 140

    provenance.consume_privacy_budget([pe2e3], 20)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 160
    provenance.consume_privacy_budget([pe2e1_2e2_1], 10)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 160
    provenance.consume_privacy_budget([pe2e2], 15)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 165

    assert pe2e1.consumed_privacy_budget == 30
    assert pe2e2.consumed_privacy_budget == 45
    assert pe2e3.consumed_privacy_budget == 40

    pe21 = provenance.new_provenance_node([pe2], "inclusive")
    pe22 = provenance.new_provenance_node([pe2], "inclusive")
    pe21_22_1 = provenance.new_provenance_node([pe21, pe22], "inclusive")
    pe2e3__21_22_1__1 = provenance.new_provenance_node([pe2e3, pe21_22_1], "inclusive")

    provenance.consume_privacy_budget([pe2e3__21_22_1__1], 10)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 175
    provenance.consume_privacy_budget([pe2e3__21_22_1__1], 10)
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 185

    pe0_ = provenance.new_provenance_root("bar")
    provenance.consume_privacy_budget([pe0_], 20)

    assert provenance.consumed_privacy_budget("bar") == pe0_.consumed_privacy_budget == 20
    assert provenance.consumed_privacy_budget("foo") == pe0.consumed_privacy_budget == 185
