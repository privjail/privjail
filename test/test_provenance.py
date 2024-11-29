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

    pe1 = provenance.new_provenance_node([pe0], "none", "inclusive")
    pe2 = provenance.new_provenance_node([pe0], "none", "inclusive")

    pe1.accumulate_privacy_budget(20)
    pe2.accumulate_privacy_budget(30)

    assert pe1.total_privacy_budget() == 20
    assert pe2.total_privacy_budget() == 30
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 60

    pe1e = provenance.new_provenance_node([pe1], "none", "exclusive")
    pe1e1 = provenance.new_provenance_node([pe1e], "none", "inclusive")
    pe1e2 = provenance.new_provenance_node([pe1e], "none", "inclusive")

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

    pe2e = provenance.new_provenance_node([pe2], "none", "exclusive")
    pe2e1 = provenance.new_provenance_node([pe2e], "none", "inclusive")
    pe2e2 = provenance.new_provenance_node([pe2e], "none", "inclusive")
    pe2e3 = provenance.new_provenance_node([pe2e], "none", "inclusive")
    pe2e1_2e2_1 = provenance.new_provenance_node([pe2e1, pe2e2], "none", "inclusive")
    pe2e1_2e3_1 = provenance.new_provenance_node([pe2e1, pe2e3], "none", "inclusive")
    pe2e2_2e3_1 = provenance.new_provenance_node([pe2e2, pe2e3], "none", "inclusive")

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

    pe21 = provenance.new_provenance_node([pe2], "none", "inclusive")
    pe22 = provenance.new_provenance_node([pe2], "none", "inclusive")
    pe21_22_1 = provenance.new_provenance_node([pe21, pe22], "none", "inclusive")
    pe2e3__21_22_1__1 = provenance.new_provenance_node([pe2e3, pe21_22_1], "none", "inclusive")

    pe2e3__21_22_1__1.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 175
    pe2e3__21_22_1__1.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 185

    pe0_ = provenance.new_provenance_root("bar")
    pe0_.accumulate_privacy_budget(20)

    assert provenance.get_privacy_budget("bar") == pe0_.total_privacy_budget() == 20
    assert provenance.get_privacy_budget("foo") == pe0.total_privacy_budget() == 185

def test_provenance_tag() -> None:
    pe0 = provenance.new_provenance_root("foo")

    pe1 = provenance.new_provenance_node([pe0], "renew", "inclusive")
    pe2 = provenance.new_provenance_node([pe0], "inherit", "inclusive")

    assert not pe0.has_same_tag(pe1)
    assert pe0.has_same_tag(pe2)
    assert not pe1.has_same_tag(pe2)

    pe1e = provenance.new_provenance_node([pe1], "inherit", "exclusive")
    pe1e1 = provenance.new_provenance_node([pe1e], "inherit", "inclusive")
    pe1e2 = provenance.new_provenance_node([pe1e], "inherit", "inclusive")

    assert pe1.has_same_tag(pe1e)
    assert pe1e.has_same_tag(pe1e1)
    assert pe1e.has_same_tag(pe1e2)
    assert pe1e1.has_same_tag(pe1e2)

    pe1e1_1e2_1 = provenance.new_provenance_node([pe1e1, pe1e2], "inherit", "inclusive")
    pe1e1_1e2_2 = provenance.new_provenance_node([pe1e1, pe1e2], "renew", "inclusive")

    assert pe1e1_1e2_1.has_same_tag(pe1e1)
    assert pe1e1_1e2_1.has_same_tag(pe1e2)
    assert not pe1e1_1e2_2.has_same_tag(pe1e1)
    assert not pe1e1_1e2_2.has_same_tag(pe1e2)
    assert not pe1e1_1e2_2.has_same_tag(pe1e1_1e2_1)

    pe1e11 = provenance.new_provenance_node([pe1e1], "none", "inclusive")
    pe21 = provenance.new_provenance_node([pe2], "none", "inclusive")
    pe1e11_21_1 = provenance.new_provenance_node([pe1e11, pe21], "none", "inclusive")

    assert pe1e11_21_1.has_same_tag(pe1e11)
    assert pe1e11_21_1.has_same_tag(pe21)

    pe0_ = provenance.new_provenance_root("bar")

    assert not pe0_.has_same_tag(pe0)
    assert not pe0_.has_same_tag(pe1)
    assert not pe0_.has_same_tag(pe2)
    assert not pe0_.has_same_tag(pe1e)
    assert not pe0_.has_same_tag(pe1e11)
