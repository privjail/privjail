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
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 0

    pe0.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 10

    pe1 = provenance.new_provenance_node([pe0], "inclusive")
    pe2 = provenance.new_provenance_node([pe0], "inclusive")

    pe1.accumulate_privacy_budget(20)
    pe2.accumulate_privacy_budget(30)

    assert pe1.total_privacy_budget() == 20
    assert pe2.total_privacy_budget() == 30
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 60

    pe1e = provenance.new_provenance_node([pe1], "exclusive")
    pe1e1 = provenance.new_provenance_node([pe1e], "inclusive")
    pe1e2 = provenance.new_provenance_node([pe1e], "inclusive")

    pe1e1.accumulate_privacy_budget(50)
    pe1e2.accumulate_privacy_budget(30)

    assert pe1e1.total_privacy_budget() == 50
    assert pe1e2.total_privacy_budget() == 30
    assert pe1e.total_privacy_budget() == 50
    assert pe1.total_privacy_budget() == 70
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 110

    pe1e2.accumulate_privacy_budget(30)

    assert pe1e1.total_privacy_budget() == 50
    assert pe1e2.total_privacy_budget() == 60
    assert pe1e.total_privacy_budget() == 60
    assert pe1.total_privacy_budget() == 80
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 120

    pe2e = provenance.new_provenance_node([pe2], "exclusive")
    pe2e1 = provenance.new_provenance_node([pe2e], "inclusive")
    pe2e2 = provenance.new_provenance_node([pe2e], "inclusive")
    pe2e3 = provenance.new_provenance_node([pe2e], "inclusive")
    pe2e1_2e2_1 = provenance.new_provenance_node([pe2e1, pe2e2], "inclusive")
    pe2e1_2e3_1 = provenance.new_provenance_node([pe2e1, pe2e3], "inclusive")
    pe2e2_2e3_1 = provenance.new_provenance_node([pe2e2, pe2e3], "inclusive")

    pe2e1_2e2_1.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 130
    pe2e2_2e3_1.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 140
    pe2e1_2e3_1.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 140

    pe2e3.accumulate_privacy_budget(20)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 160
    pe2e1_2e2_1.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 160
    pe2e2.accumulate_privacy_budget(15)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 165

    assert pe2e1.total_privacy_budget() == 30
    assert pe2e2.total_privacy_budget() == 45
    assert pe2e3.total_privacy_budget() == 40

    pe21 = provenance.new_provenance_node([pe2], "inclusive")
    pe22 = provenance.new_provenance_node([pe2], "inclusive")
    pe21_22_1 = provenance.new_provenance_node([pe21, pe22], "inclusive")
    pe2e3__21_22_1__1 = provenance.new_provenance_node([pe2e3, pe21_22_1], "inclusive")

    pe2e3__21_22_1__1.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 175
    pe2e3__21_22_1__1.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 185

    pe0_ = provenance.new_provenance_root("bar")
    pe0_.accumulate_privacy_budget(20)

    assert provenance.get_privacy_budget("bar") == pe0_.total_privacy_budget() == 20
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 185
