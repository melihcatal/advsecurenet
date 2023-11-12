"""
This module contains utility functions for working with configuration data through the CLI.
"""
import os
import yaml
import pkg_resources
from dataclasses import fields
from typing import Dict, Callable, TypeVar, Type
from advsecurenet.shared.types import ConfigType

config_path = pkg_resources.resource_filename("advsecurenet", "configs")
T = TypeVar('T')


def build_config(config_data, config_type):
    """
    Build a configuration object from the provided configuration data.
    """
    expected_keys = [f.name for f in fields(config_type)]
    filtered_config_data = {k: config_data[k]
                            for k in expected_keys if k in config_data}
    return config_type(**filtered_config_data)


def read_config_file(config_file: str):
    """Reads the configuration file."""
    try:
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at {config_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid configuration file: {e}")


def load_configuration(config_type: ConfigType, config_file: str, **overrides: Dict):
    """Loads and overrides the configuration."""
    # Load the base configuration
    config_data = read_config_file(config_file)
    # Call specific checks
    handler = CHECK_HANDLERS.get(config_type)
    if handler:
        handler(config_data, overrides)

    # Override the base config with the provided overrides if not None
    config_data.update({k: v for k, v in overrides.items() if v is not None})
    return config_data


def attack_config_check(config_data: Dict, overrides: Dict):
    if overrides.get('dataset_name') == 'custom' and not overrides.get('custom_data_dir'):
        raise ValueError(
            "Please provide a valid path for custom-data-dir when using the custom dataset.")
    if overrides.get('dataset_name') != 'custom' and overrides.get('custom_data_dir'):
        raise ValueError(
            "Please set dataset-name to 'custom' when specifying custom-data-dir.")


CHECK_HANDLERS = {
    ConfigType.ATTACK: attack_config_check,
}
