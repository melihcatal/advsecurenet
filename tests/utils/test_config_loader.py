import pytest
import yaml
import os
from dataclasses import dataclass, asdict, field
from advsecurenet.utils.config_loader import load_config_from_yaml

# This is a temporary file, so its path should be unique
TEMP_YAML_PATH = "/tmp/test_config.yaml"

@dataclass
class SampleConfig:
    name: str = "default_name"
    age: int = 25
    items: list = field(default_factory=list)

@pytest.fixture
def sample_yaml_content():
    content = """
    name: test_name
    age: 30
    items:
      - item1
      - item2
    """
    with open(TEMP_YAML_PATH, 'w') as f:
        f.write(content)
    yield TEMP_YAML_PATH
    # Clean up after the test
    os.remove(TEMP_YAML_PATH)

def test_load_config_from_yaml(sample_yaml_content):
    config = load_config_from_yaml(sample_yaml_content, SampleConfig)

    assert config.name == "test_name"
    assert config.age == 30
    assert config.items == ["item1", "item2"]
