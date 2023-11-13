import pytest
import os
import tempfile
from dataclasses import dataclass
from advsecurenet.utils.config_loader import load_config_from_yaml
from advsecurenet.utils.config_utils import get_config_file_name, save_config_to_yaml, load_default_config, generate_default_config_yaml, get_available_configs, read_yml_file, get_default_config_yml, override_with_cli_args


# Sample DataClass for testing
@dataclass
class SampleConfig:
    a: int = 0
    b: str = "test"


# Path to a temporary YAML file for testing
temp_yaml_path = "temp_config.yaml"

# Clean up temporary file after tests


def teardown_module():
    if os.path.exists(temp_yaml_path):
        os.remove(temp_yaml_path)


def test_load_config_from_yaml():
    # Given a sample YAML file
    with open(temp_yaml_path, 'w') as f:
        f.write("a: 5\nb: hello")

    # When loading this YAML into the SampleConfig dataclass
    result = load_config_from_yaml(temp_yaml_path, SampleConfig)

    # Then the values should be correctly loaded
    assert result.a == 5
    assert result.b == "hello"


def test_save_config_to_yaml():
    # Given a dataclass instance
    config = SampleConfig(a=7, b="world")

    # When saving this dataclass to a YAML file
    save_config_to_yaml(config, temp_yaml_path)

    # Then the file should contain the correct YAML
    with open(temp_yaml_path, 'r') as f:
        content = f.read()

    assert content.strip() == "a: 7\nb: world"


def test_get_config_file_name():
    # Given the SampleConfig dataclass
    # When getting the config file name for it
    result = get_config_file_name(SampleConfig)

    # Then the result should be the snake_case version of the class name
    assert result == "sample_config.yml"


@dataclass
class AnotherSampleConfig:
    x: int = 10
    y: str = "sample"
    z: list = None


def test_load_default_config(mocker):
    # Mocking the resource_filename to return a temporary path
    mocker.patch("pkg_resources.resource_filename",
                 return_value=tempfile.gettempdir())

    # Creating a temporary default config
    with open(os.path.join(tempfile.gettempdir(), "another_sample_config.yml"), 'w') as f:
        f.write("x: 20\ny: default")

    result = load_default_config(AnotherSampleConfig)
    assert result.x == 20
    assert result.y == "default"
    assert result.z is None


def test_generate_default_config_yaml(mocker):
    # Mocking the resource_filename to return a temporary path
    mocker.patch("pkg_resources.resource_filename",
                 return_value=tempfile.gettempdir())

    # Creating a temporary default config
    temp_default_path = os.path.join(
        tempfile.gettempdir(), "another_sample_config.yml")
    with open(temp_default_path, 'w') as f:
        f.write("x: 25\ny: generated")

    # Mocking get_default_config_yml to return the temp_default_path
    mocker.patch("advsecurenet.utils.config_utils.get_default_config_yml",
                 return_value=temp_default_path)

    with tempfile.TemporaryDirectory() as tempdir:
        result = generate_default_config_yaml(
            "another_sample_config.yml", tempdir, save=True)
        assert result == {"x": 25, "y": "generated"}

        with open(os.path.join(tempdir, "another_sample_config.yml"), 'r') as f:
            content = f.read()
        assert content is not None, "Content read from the file is None"
        assert "x: 25" in content
        assert "y: generated" in content


def test_get_available_configs(mocker):
    # Mock the pkg_resources.resource_filename to return a temporary path
    temp_dir = tempfile.gettempdir()
    mocker.patch("pkg_resources.resource_filename", return_value=temp_dir)

    # Assuming the actual structure of your config directory is configs/cli
    cli_configs_dir = os.path.join(temp_dir, "configs/cli")
    # Create the mock directory structure
    os.makedirs(cli_configs_dir, exist_ok=True)

    # Create mock config files in the temporary directory
    mock_files = ["lots_attack_config.yml",
                  "cw_attack_config.yml", "some_other_file.txt"]
    for file_name in mock_files:
        with open(os.path.join(cli_configs_dir, file_name), 'w') as f:
            f.write("mock data")

    # Mock os.listdir to return our mock_files
    mocker.patch("os.listdir", return_value=mock_files)

    configs = get_available_configs()

    # Clean up the temporary files created for the test
    for file_name in mock_files:
        os.remove(os.path.join(cli_configs_dir, file_name))

    assert "lots_attack_config.yml" in configs
    assert "cw_attack_config.yml" in configs
    assert "some_other_file.txt" not in configs


def test_read_yml_file():
    # Creating a temporary YAML file
    with tempfile.NamedTemporaryFile('w', delete=False, suffix=".yml") as temp:
        temp.write("x: 50\ny: temp_file")
        temp_path = temp.name

    result = read_yml_file(temp_path)
    assert result == {"x": 50, "y": "temp_file"}


def test_get_default_config_yml(mocker):
    # Define a mock config_path
    mock_config_path = tempfile.gettempdir()

    # Mocking config_path directly
    mocker.patch("advsecurenet.utils.config_utils.config_path",
                 mock_config_path)

    # Creating a temporary default config
    temp_default_path = os.path.join(mock_config_path, "desired_config.yml")
    with open(temp_default_path, 'w') as f:
        f.write("x: 60\ny: desired")

    result_path = get_default_config_yml("desired_config.yml")
    assert result_path is not None, "Result path is None"
    assert result_path == temp_default_path


def test_override_with_cli_args():
    config_data = {"x": 10, "y": "initial"}
    result = override_with_cli_args(config_data, x=20, z="new")
    assert result == {"x": 20, "y": "initial", "z": "new"}
