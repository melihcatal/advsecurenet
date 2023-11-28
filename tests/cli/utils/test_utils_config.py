from dataclasses import dataclass
import pytest
from yaml.scanner import ScannerError
from cli.utils.config import build_config, read_config_file, load_configuration, attack_config_check
from advsecurenet.shared.types.configs.configs import ConfigType

# Mock configuration class for testing build_config


@dataclass
class MockConfig:
    field1: str
    field2: int


def test_build_config():
    data = {
        'field1': 'test',
        'field2': 2,
        'field3': 'not_needed'
    }
    config = build_config(data, MockConfig)
    assert config.field1 == 'test'
    assert config.field2 == 2
    with pytest.raises(AttributeError):
        _ = config.field3


def test_read_config_file(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "config.yml"
    p.write_text("field1: test\nfield2: 2\n")

    config_data = read_config_file(p)
    assert config_data['field1'] == 'test'
    assert config_data['field2'] == 2

    with pytest.raises(FileNotFoundError):
        read_config_file("/nonexistent/config.yml")

    bad_yaml = d / "bad_config.yml"
    bad_yaml.write_text("field1: test\nfield2=4\n")
    with pytest.raises(ScannerError):
        read_config_file(bad_yaml)


def test_load_configuration(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "config.yml"
    p.write_text("field1: test\nfield2: 2\n")

    data = load_configuration(ConfigType.ATTACK, p,
                              field1="overridden", field2=3)
    assert data['field1'] == 'overridden'
    assert data['field2'] == 3


def test_attack_config_check():
    config_data = {}
    overrides = {
        'dataset_name': 'custom',
        'custom_data_dir': None
    }
    with pytest.raises(ValueError, match="Please provide a valid path for custom-data-dir when using the custom dataset."):
        attack_config_check(overrides)

    overrides['dataset_name'] = 'not_custom'
    overrides['custom_data_dir'] = 'test'
    with pytest.raises(ValueError, match="Please set dataset-name to 'custom' when specifying custom-data-dir."):
        attack_config_check(overrides)
