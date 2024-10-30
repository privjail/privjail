from typing import Generator
import pytest
import importlib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
provenance = importlib.import_module("pripri.provenance")

@pytest.fixture(autouse=True)
def setup() -> Generator[None]:
    provenance.clear_global_states()
    yield

def test_provenance_accumulation() -> None:
    pe0 = provenance.new_provenance_root("foo")
    assert provenance.get_privacy_budget("foo") == pe0.privacy_budget == 0

    pe0.accumulate_privacy_budget(10)
    assert provenance.get_privacy_budget("foo") == pe0.privacy_budget == 10

    pe1 = pe0.add_child(children_type="inclusive")
    pe2 = pe0.add_child(children_type="inclusive")

    pe1.accumulate_privacy_budget(20)
    pe2.accumulate_privacy_budget(30)

    assert pe1.privacy_budget == 20
    assert pe2.privacy_budget == 30
    assert provenance.get_privacy_budget("foo") == pe0.privacy_budget == 60

    pe1e = pe1.add_child(children_type="exclusive")
    pe1e1 = pe1e.add_child(children_type="inclusive")
    pe1e2 = pe1e.add_child(children_type="inclusive")

    pe1e1.accumulate_privacy_budget(50)
    pe1e2.accumulate_privacy_budget(30)

    assert pe1e1.privacy_budget == 50
    assert pe1e2.privacy_budget == 30
    assert pe1e.privacy_budget == 50
    assert pe1.privacy_budget == 70
    assert provenance.get_privacy_budget("foo") == pe0.privacy_budget == 110

    pe1e2.accumulate_privacy_budget(30)

    assert pe1e1.privacy_budget == 50
    assert pe1e2.privacy_budget == 60
    assert pe1e.privacy_budget == 60
    assert pe1.privacy_budget == 80
    assert provenance.get_privacy_budget("foo") == pe0.privacy_budget == 120

    pe0_ = provenance.new_provenance_root("bar")
    pe0_.accumulate_privacy_budget(20)

    assert provenance.get_privacy_budget("bar") == pe0_.privacy_budget == 20
    assert provenance.get_privacy_budget("foo") == pe0.privacy_budget == 120

def test_provenance_tag() -> None:
    pe0 = provenance.new_provenance_root("foo")

    pe1 = pe0.add_child(children_type="inclusive")
    pe2 = pe0.add_child(children_type="inclusive")

    assert provenance.have_same_tag(pe0, pe1)
    assert provenance.have_same_tag(pe0, pe2)
    assert provenance.have_same_tag(pe1, pe2)

    pe1.new_tag()

    assert not provenance.have_same_tag(pe0, pe1)
    assert provenance.have_same_tag(pe0, pe2)
    assert not provenance.have_same_tag(pe1, pe2)

    pe2.new_tag()

    assert not provenance.have_same_tag(pe0, pe1)
    assert not provenance.have_same_tag(pe0, pe2)
    assert not provenance.have_same_tag(pe1, pe2)

    pe1e = pe1.add_child(children_type="exclusive")
    pe1e1 = pe1e.add_child(children_type="inclusive")
    pe1e2 = pe1e.add_child(children_type="inclusive")

    assert provenance.have_same_tag(pe1, pe1e)
    assert provenance.have_same_tag(pe1e, pe1e1)
    assert provenance.have_same_tag(pe1e, pe1e2)
    assert provenance.have_same_tag(pe1e1, pe1e2)

    pe0_ = provenance.new_provenance_root("bar")

    assert not provenance.have_same_tag(pe0_, pe0)
    assert not provenance.have_same_tag(pe0_, pe1)
    assert not provenance.have_same_tag(pe0_, pe2)
    assert not provenance.have_same_tag(pe0_, pe1e)
