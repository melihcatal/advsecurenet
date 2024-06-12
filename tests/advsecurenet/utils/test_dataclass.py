from dataclasses import dataclass
from typing import List, Optional, Union

import pytest

from advsecurenet.utils.dataclass import (filter_for_dataclass,
                                          flatten_dataclass, merge_dataclasses,
                                          recursive_dataclass_instantiation)


@dataclass
class Nested:
    value: int


@dataclass
class Example:
    a: int
    b: str
    c: Nested
    d: Optional[int] = None
    e: Optional[Nested] = None


@dataclass
class AnotherExample:
    a: int
    b: str
    f: float


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_flatten_dataclass():
    instance = Example(a=1, b="test", c=Nested(value=10))
    flattened = flatten_dataclass(instance)
    assert flattened == {
        "a": 1,
        "b": "test",
        "c": {"value": 10},
        "d": None,
        "e": None
    }


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_filter_for_dataclass():
    data = {
        "a": 1,
        "b": "test",
        "c": {"value": 10},
        "extra_field": "extra"
    }
    filtered = filter_for_dataclass(data, Example)
    assert filtered == {
        "a": 1,
        "b": "test",
        "c": {"value": 10}
    }


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_filter_for_dataclass_with_convert():
    data = {
        "a": 1,
        "b": "test",
        "c": {"value": 10},
        "extra_field": "extra"
    }
    filtered = filter_for_dataclass(data, Example, convert=True)
    assert isinstance(filtered, Example)
    assert filtered.a == 1
    assert filtered.b == "test"
    assert filtered.c.value == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_recursive_dataclass_instantiation():
    data = {
        "a": 1,
        "b": "test",
        "c": {"value": 10}
    }
    instance = recursive_dataclass_instantiation(Example, data)
    assert isinstance(instance, Example)
    assert instance.a == 1
    assert instance.b == "test"
    assert instance.c.value == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_merge_dataclasses():
    dataclass1 = Example(a=1, b="test", c=Nested(value=10))
    dataclass2 = AnotherExample(a=2, b="updated", f=3.14)
    merged = merge_dataclasses(dataclass1, dataclass2)
    assert isinstance(merged, Example)
    assert merged.a == 2
    assert merged.b == "updated"
    assert merged.c.value == 10
    assert not hasattr(merged, "f")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_merge_dataclasses_with_optional():
    dataclass1 = Example(a=1, b="test", c=Nested(value=10), d=5)
    dataclass2 = Example(
        a=2,
        b="updated",
          c=Nested(
              value=20), e=Nested(value=30))
    merged = merge_dataclasses(dataclass1, dataclass2)
    assert isinstance(merged, Example)
    assert merged.a == 2
    assert merged.b == "updated"
    assert merged.c.value == 20
    assert merged.e.value == 30
