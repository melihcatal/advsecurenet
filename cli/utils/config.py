"""
This module contains utility functions for working with configuration data through the CLI.
"""
import re
from dataclasses import fields
from typing import Any, Dict, Type, TypeVar

import click
import pkg_resources
import yaml

from advsecurenet.shared.types.configs.configs import ConfigType
from advsecurenet.utils.config_utils import (get_default_config_yml,
                                             read_yml_file)
from advsecurenet.utils.dataclass import recursive_dataclass_instantiation

config_path = pkg_resources.resource_filename("advsecurenet", "configs")
# This will represent the specific class type of `config_class`
T = TypeVar('T')


def build_config(config_data, config_type):
    """
    Build a configuration object from the provided configuration data.
    """
    expected_keys = [f.name for f in fields(config_type)]
    filtered_config_data = {k: config_data[k]
                            for k in expected_keys if k in config_data}
    return config_type(**filtered_config_data)


def read_config_file(config_file: str) -> Dict:
    """Reads the configuration file.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        Dict: The configuration data.

    """
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
        return config_data


def load_configuration(config_type: ConfigType, config_file: str, **overrides: Dict):
    """Loads and overrides the configuration."""
    # Load the base configuration
    config_data = read_yml_file(config_file)
    # Call specific checks
    handler = CHECK_HANDLERS.get(config_type)
    if handler:
        handler(overrides)

    # Override the base config with the provided overrides if not None
    config_data.update({k: v for k, v in overrides.items() if v is not None})
    return config_data


def attack_config_check(overrides: Dict):
    """
    Checks the configuration for the attack command.

    Args:
        overrides (Dict): The overrides provided through the CLI.

    Raises:
        ValueError: If the dataset name is not 'custom' when specifying a custom data directory.
        ValueError: If the dataset name is 'custom' when not specifying a custom data directory.
    """

    if overrides.get('dataset_name') == 'custom' and not overrides.get('custom_data_dir'):
        raise ValueError(
            "Please provide a valid path for custom-data-dir when using the custom dataset.")
    if overrides.get('dataset_name') != 'custom' and overrides.get('custom_data_dir'):
        raise ValueError(
            "Please set dataset-name to 'custom' when specifying custom-data-dir.")


CHECK_HANDLERS = {
    ConfigType.ATTACK: attack_config_check,
}


def load_yaml_with_include(file_path: str
                           ):
    """
    Load a YAML file with support for includes.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = f.read()
        # Pre-process includes
        contents = re.sub(r'!include (.+\.yaml)', r'!include \1', contents)
        return yaml.load(contents, Loader=yaml.FullLoader)


def load_and_instantiate_config(config: str, default_config_file: str, config_type: ConfigType, config_class: Type[T], **kwargs: Dict[str, Any]) -> T:
    """Utility function to load and instantiate configuration.

    Args:
        config (str): The path to the configuration file.
        default_config_file (str): The default configuration file name.
        config_type (ConfigType): The type of configuration.
        config_class (Type[T]): The dataclass type for configuration instantiation.
        **kwargs (Dict[str, Any]): Additional keyword arguments. If provided, they will override the configuration.

    Returns:
        T: An instantiated configuration data class of the type specified by `config_class`.
    """
    if not config:
        click.secho(
            "No configuration file provided! Using default configuration...", fg="blue")
        config = get_default_config_yml(default_config_file, "cli")

    config_data = load_configuration(
        config_type=config_type, config_file=config, **kwargs)
    return recursive_dataclass_instantiation(config_class, config_data)
