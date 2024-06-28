import logging
import os
from dataclasses import dataclass
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch

import pytest

from advsecurenet.shared.types.configs.configs import ConfigType
from cli.shared.utils.config import (
    _include_yaml,
    attack_config_check,
    build_config,
    deep_update,
    generate_default_config_yaml,
    get_available_configs,
    get_default_config_yml,
    load_and_instantiate_config,
    load_configuration,
    make_paths_absolute,
    read_yml_file,
)

logger = logging.getLogger("cli.shared.utils.config")


# Mock data for tests
yaml_content = """
key: value
"""


@dataclass
class SampleConfig:
    field1: int
    field2: str


@pytest.mark.cli
@pytest.mark.essential
def test_build_config():
    config_data = {"field1": 10, "field2": "value", "extra_field": "extra"}
    config = build_config(config_data, SampleConfig)
    assert config.field1 == 10
    assert config.field2 == "value"
    assert not hasattr(config, "extra_field")


@pytest.mark.cli
@pytest.mark.essential
def test_deep_update():
    source = {"a": 1, "b": {"c": 2, "d": 3}, "e": 10}
    overrides = {"b": {"c": 3, "d": 4}, "e": 5}
    updated = deep_update(source, overrides)
    assert updated == {"a": 1, "b": {"c": 3, "d": 4}, "e": 5}


@pytest.mark.cli
@pytest.mark.essential
@patch(
    "cli.shared.utils.config.read_yml_file",
    return_value={"key": "value", "override_key": "override_initial_value"},
)
def test_load_configuration(mock_read_yml_file):
    config_file = "test.yml"
    overrides = {"override_key": "override_value"}
    config_type = ConfigType.ATTACK

    result = load_configuration(config_type, config_file, **overrides)
    mock_read_yml_file.assert_called_once_with(config_file)
    assert result == {"key": "value", "override_key": "override_value"}


@pytest.mark.cli
@pytest.mark.essential
def test_attack_config_check():
    with pytest.raises(
        ValueError,
        match="Please provide a valid path for custom-data-dir when using the custom dataset.",
    ):
        attack_config_check({"dataset_name": "custom"})
    with pytest.raises(
        ValueError,
        match="Please set dataset-name to 'custom' when specifying custom-data-dir.",
    ):
        attack_config_check(
            {"dataset_name": "non_custom", "custom_data_dir": "some_dir"}
        )


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.config.click.secho")
@patch(
    "cli.shared.utils.config.get_default_config_yml", return_value="default_config.yml"
)
@patch("cli.shared.utils.config.load_configuration", return_value={"key": "value"})
@patch(
    "cli.shared.utils.config.recursive_dataclass_instantiation",
    return_value=SampleConfig(field1=1, field2="test"),
)
def test_load_and_instantiate_config(
    mock_recursive_instantiation,
    mock_load_configuration,
    mock_get_default_config_yml,
    mock_click_secho,
):
    config = ""
    default_config_file = "default.yml"
    config_type = ConfigType.ATTACK
    config_class = SampleConfig

    result = load_and_instantiate_config(
        config, default_config_file, config_type, config_class
    )
    mock_click_secho.assert_called_once()
    mock_get_default_config_yml.assert_called_once_with(default_config_file)
    mock_load_configuration.assert_called_once_with(
        config_type=config_type, config_file="default_config.yml"
    )
    mock_recursive_instantiation.assert_called_once_with(config_class, {"key": "value"})
    assert result == SampleConfig(field1=1, field2="test")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.config.os.walk")
@patch(
    "cli.shared.utils.config.os.path.exists", side_effect=Exception("Test Exception")
)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="MODULE_TITLE = 'Title'\nMODULE_DESCRIPTION = 'Description'\nINCLUDE_IN_CLI_CONFIGS = True",
)
def test_get_available_configs_exception(mock_open, mock_path_exists, mock_os_walk):
    mock_os_walk.return_value = [
        ("dirpath", None, ["file1_config.yml", "file2_config.yml"])
    ]

    with mock.patch.object(logger, "error") as mock_logging_error:
        result = get_available_configs()
        mock_logging_error.assert_called_once()
        assert result == []


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.config.os.walk")
@patch("cli.shared.utils.config.os.path.exists", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data="MODULE_TITLE = 'Title'\nMODULE_DESCRIPTION = 'Description'\nINCLUDE_IN_CLI_CONFIGS = True",
)
def test_get_available_configs(mock_open, mock_path_exists, mock_os_walk):
    mock_os_walk.return_value = [
        ("dirpath", None, ["file1_config.yml", "file2_config.yml"])
    ]

    result = get_available_configs()
    assert len(result) == 2
    assert result[0]["title"] == "Title"
    assert result[0]["description"] == "Description"


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.config.os.path.exists", return_value=True)
@patch("cli.shared.utils.config.os.makedirs")
@patch("cli.shared.utils.config.read_yml_file", return_value={"key": "value"})
@patch("builtins.open", new_callable=mock_open)
@patch("cli.shared.utils.config.YAML")
def test_generate_default_config_yaml(
    mock_yaml, mock_open, mock_read_yml_file, mock_makedirs, mock_path_exists
):
    mock_yaml_instance = mock_yaml.return_value
    config_name = "test_config.yml"
    output_path = "output.yml"

    result = generate_default_config_yaml(config_name, output_path, save=True)
    mock_read_yml_file.assert_called_once()
    mock_open.assert_called_once_with(output_path, "w+", encoding="utf-8")
    mock_yaml_instance.dump.assert_called_once_with({"key": "value"}, mock_open())
    assert result == {"key": "value"}


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data="key: value")
def test_read_yml_file(mock_open):
    result = read_yml_file("test.yml")
    assert result == {"key": "value"}
    mock_open.assert_called_once_with("test.yml", "r", encoding="utf-8")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.config.os.path.exists", return_value=True)
@patch(
    "cli.shared.utils.config.os.walk",
    return_value=[("dirpath", None, ["test_config.yml"])],
)
def test_get_default_config_yml(mock_os_walk, mock_path_exists):
    config_name = "test_config.yml"
    result = get_default_config_yml(config_name)
    assert result == "dirpath/test_config.yml"


@pytest.fixture
def loader():
    loader = MagicMock()
    loader.name = "/path/to/current/file.yaml"
    return loader


@pytest.fixture
def node():
    node = MagicMock()
    node.value = "included.yaml"
    return node


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", new_callable=mock_open, read_data=yaml_content)
@patch("yaml.load", return_value={"key": "value"})
def test_include_yaml_success(mock_yaml_load, mock_file_open, loader, node):
    result = _include_yaml(loader, node)

    mock_file_open.assert_called_once_with(
        "/path/to/current/included.yaml", "r", encoding="utf-8"
    )
    mock_yaml_load.assert_called_once()
    assert result == {"key": "value"}


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", side_effect=FileNotFoundError)
def test_include_yaml_file_not_found(mock_file_open, loader, node):
    with mock.patch.object(logger, "error") as mock_logging_error:
        result = _include_yaml(loader, node)

        mock_file_open.assert_called_once_with(
            "/path/to/current/included.yaml", "r", encoding="utf-8"
        )
        mock_logging_error.assert_called_once()
        assert result is None


@pytest.mark.cli
@pytest.mark.essential
@patch("builtins.open", side_effect=Exception("Test Exception"))
def test_include_yaml_general_exception(mock_file_open, loader, node):
    with mock.patch.object(logger, "error") as mock_logging_error:
        result = _include_yaml(loader, node)

        mock_file_open.assert_called_once_with(
            "/path/to/current/included.yaml", "r", encoding="utf-8"
        )
        mock_logging_error.assert_called_once_with(
            "Error loading file: %s", "Test Exception"
        )
        assert result is None


@pytest.fixture
def base_path():
    return "/base/path"


@pytest.fixture
def config_dict():
    return {
        "relative_path": "relative/path/to/file",
        "absolute_path": "/absolute/path/to/file",
        "nested": {
            "relative_dir": "relative/path/to/dir",
            "not_a_path": "just_a_string",
        },
    }


@pytest.fixture
def config_list():
    return [
        {"relative_path": "relative/path/to/file"},
        {"absolute_path": "/absolute/path/to/file"},
        {"nested": {"relative_dir": "relative/path/to/dir"}},
    ]


@patch("os.path.exists", return_value=True)
@patch("os.path.abspath", side_effect=lambda x: os.path.join("/absolute", x.strip("/")))
def test_make_paths_absolute_dict(mock_abspath, mock_exists, base_path, config_dict):
    make_paths_absolute(base_path, config_dict)

    assert config_dict["relative_path"] == "/absolute/base/path/relative/path/to/file"
    assert config_dict["absolute_path"] == "/absolute/path/to/file"
    assert (
        config_dict["nested"]["relative_dir"]
        == "/absolute/base/path/relative/path/to/dir"
    )


@patch("os.path.exists", return_value=True)
@patch("os.path.abspath", side_effect=lambda x: os.path.join("/absolute", x.strip("/")))
def test_make_paths_absolute_list(mock_abspath, mock_exists, base_path, config_list):
    make_paths_absolute(base_path, config_list)

    assert (
        config_list[0]["relative_path"] == "/absolute/base/path/relative/path/to/file"
    )
    assert config_list[1]["absolute_path"] == "/absolute/path/to/file"
    assert (
        config_list[2]["nested"]["relative_dir"]
        == "/absolute/base/path/relative/path/to/dir"
    )


@patch("os.path.exists", side_effect=lambda x: x.endswith("file"))
@patch("os.path.abspath", side_effect=lambda x: os.path.join("/absolute", x.strip("/")))
def test_make_paths_absolute_mixed(mock_abspath, mock_exists, base_path, config_dict):
    config_dict["file_check"] = "relative/path/to/file"
    config_dict["dir_check"] = "relative/path/to/dir"
    config_dict["should_not_fix"] = "should_not_fix"

    make_paths_absolute(base_path, config_dict)

    assert config_dict["file_check"] == "/absolute/base/path/relative/path/to/file"
    assert config_dict["dir_check"] == "/absolute/base/path/relative/path/to/dir"
    assert config_dict["should_not_fix"] == "should_not_fix"
