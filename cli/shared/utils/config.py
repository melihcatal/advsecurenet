"""
This module contains utility functions for working with configuration data through the CLI.
"""
import logging
import os
from dataclasses import fields
from typing import Any, Dict, List, Type, TypeVar, Union

import click
import pkg_resources
import yaml
from ruamel.yaml import YAML

from advsecurenet.shared.types.configs.configs import ConfigType
from advsecurenet.utils.dataclass import recursive_dataclass_instantiation

config_path = pkg_resources.resource_filename("cli", "configs")
CONFIG_FILE_ENDING = "_config.yml"
# This will represent the specific class type of `config_class`
T = TypeVar('T')
logger = logging.getLogger(__name__)


def build_config(config_data: Dict,
                 config_type: Type[T]) -> T:
    """
    Build a configuration object from the provided configuration data.

    Parameters:
    - config_data (Dict): The configuration data.
    - config_type (Type[T]): The type of configuration object to build.

    Returns:
    - T: The configuration object of the specified type.

    """
    expected_keys = [f.name for f in fields(config_type)]
    filtered_config_data = {k: config_data[k]
                            for k in expected_keys if k in config_data}
    return config_type(**filtered_config_data)


def deep_update(source: Dict, overrides: Dict) -> Dict:
    """
    Recursively update a dictionary by overriding its keys.

    Parameters:
    - source (Dict): The original dictionary with the base values.
    - overrides (Dict): The dictionary with overrides, where the keys might be nested.

    Returns:
    - Dict: The updated dictionary.
    """
    for key, value in source.items():
        if key in overrides:
            if isinstance(value, dict) and isinstance(overrides[key], dict):
                # Recursively update dictionaries.
                source[key] = deep_update(value, overrides[key])
            elif overrides[key] is not None:
                # Update only if the override value is not None.
                source[key] = overrides[key]
        elif isinstance(value, dict):
            # Continue to search for deeper overrides.
            source[key] = deep_update(value, overrides)
    return source


def load_configuration(config_type: ConfigType, config_file: str, **overrides: Dict) -> Dict:
    """
    Loads and overrides the configuration.

    Parameters:
    - config_type (ConfigType): The type of configuration.
    - config_file (str): The path to the configuration file.
    - **overrides (Dict): The overrides to apply to the configuration.

    Returns:
    - Dict: The configuration data.
    """
    # Load the base configuration
    config_data = read_yml_file(config_file)
    # Call specific checks
    handler = CHECK_HANDLERS.get(config_type)
    if handler:
        handler(overrides)
    # Deeply override the base config with the provided overrides
    updated_config = deep_update(config_data, overrides)
    return updated_config


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


def is_path_to_update(key: str, value: str, base_path: str) -> bool:
    """
    Determine if the path needs to be updated to an absolute path.
    Args:
        key (str): The key in the configuration dictionary.
        value (str): The value to check.
        base_path (str): The base directory path.
    Returns:
        bool: True if the path should be updated, False otherwise.
    """
    return (
        isinstance(value, str) and not os.path.isabs(value) and
        (os.path.exists(os.path.join(base_path, value))
         or "dir" in key or "path" in key)
    )


def update_dict_paths(base_path: str, config: Dict[str, Any]) -> None:
    """
    Update paths in a dictionary configuration.
    Args:
        base_path (str): The base directory path.
        config (Dict[str, Any]): The dictionary configuration data to update.
    """
    for key, value in config.items():
        if is_path_to_update(key, value, base_path):
            config[key] = os.path.abspath(os.path.join(base_path, value))
        elif isinstance(value, (dict, list)):
            make_paths_absolute(base_path, value)


def update_list_paths(base_path: str, config: List[Any]) -> None:
    """
    Update paths in a list configuration.
    Args:
        base_path (str): The base directory path.
        config (List[Any]): The list configuration data to update.
    """
    for item in config:
        make_paths_absolute(base_path, item)


def make_paths_absolute(base_path: str, config: Union[Dict[str, Any], List[Any]]) -> None:
    """
    Recursively update the paths in the configuration dictionary or list to be absolute paths.
    Args:
        base_path (str): The base directory path.
        config (Union[Dict[str, Any], List[Any]]): The configuration data to update.
    """
    if isinstance(config, dict):
        update_dict_paths(base_path, config)
    elif isinstance(config, list):
        update_list_paths(base_path, config)


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
        config = get_default_config_yml(default_config_file)

    # Load the configuration data
    config_data = load_configuration(
        config_type=config_type, config_file=config, **kwargs)

    # Convert relative paths to absolute paths
    base_path = os.path.dirname(config)
    make_paths_absolute(base_path, config_data)

    return recursive_dataclass_instantiation(config_class, config_data)


def get_available_configs() -> list:
    """
    Get a list of all available configuration settings, including those in subdirectories,
    based on attributes defined in each module's __init__.py file.

    Parameters:
        config_path (str): The base directory path where the configuration files are stored.

    Returns:
        list: A list of dictionaries with the configuration settings including custom title, description,
        and config file name, filtered by the INCLUDE_IN_CLI_CONFIGS attribute.

    Examples:
        >>> from advsecurenet.utils.config_utils import get_available_configs
        >>> get_available_configs('/path/to/configs')
        [
            {'title': 'Example Configuration Module', 'description': 'Configuration for attack types.', 'config_file': 'lots_attack_config.yml'},
            {'title': 'Example Configuration Module', 'description': 'Configuration for defense mechanisms.', 'config_file': 'cw_attack_config.yml'}
        ]
    """
    configs = []
    for dirpath, _, files in os.walk(config_path):
        init_file_path = os.path.join(dirpath, "__init__.py")
        title = os.path.basename(dirpath)  # Default to folder name as title
        description = "No description provided"  # Default description
        include_in_cli = True  # Default to include if the flag is not specified
        try:
            # Evaluate __init__.py for title, description and INCLUDE_IN_CLI_CONFIGS
            if os.path.exists(init_file_path):
                exec_namespace = {}
                with open(init_file_path, 'r', encoding='utf-8') as f:
                    exec_content = f.read()
                    exec(exec_content, {}, exec_namespace)
                    # Use custom title if provided
                    title = exec_namespace.get('MODULE_TITLE', title).strip()
                    description = exec_namespace.get(
                        'MODULE_DESCRIPTION', description).strip()
                    include_in_cli = exec_namespace.get(
                        'INCLUDE_IN_CLI_CONFIGS', include_in_cli)
            if include_in_cli:
                config_files = [
                    file for file in files if file.endswith(CONFIG_FILE_ENDING)]
                for config_file in config_files:
                    configs.append({
                        "title": title,
                        "description": description,
                        "config_file": config_file
                    })
        except Exception as e:
            logger.error(
                "Error occurred while reading configuration files: %s", e)
            return []

    return configs


def generate_default_config_yaml(config_name: str, output_path: str, save=False, config_subdir=None) -> dict:
    """
    Generate a default configuration YAML based on the name of the configuration.

    Args:
        config_name (str): The name of the configuration yml file (e.g., "cw_attack_config.yml").
        output_path (str): The path where the YAML file should be saved.

    Returns:
        dict: The default configuration YAML as a dictionary.
    """
    if config_name is None or output_path is None:
        raise ValueError(
            "config_name and output_path must be specified and not None!")

    if not config_name.endswith(CONFIG_FILE_ENDING):
        config_name = config_name + CONFIG_FILE_ENDING

    default_config_path = get_default_config_yml(config_name, config_subdir)
    # if the file does not exist, raise an error
    if not os.path.exists(default_config_path):
        raise FileNotFoundError("The default config file does not exist!")

    default_config = read_yml_file(default_config_path)
    if not output_path.endswith(".yml"):
        # If it's not a .yml file, assume it's a directory and append the filename
        output_path = os.path.join(output_path, config_name)

    # if the directory does not exist, create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # create the yml file
    if save:
        yaml_obj = YAML()
        print(f"Saving default config to {output_path}")
        # if the file does not exist, create it otherwise overwrite it
        with open(output_path, 'w+', encoding='utf-8') as file:
            yaml_obj.dump(default_config, file)
            # yaml.dump(default_config, file)

    return default_config


def _include_yaml(loader: yaml.Loader, node: yaml.Node) -> Union[None, Dict]:
    """ 
    Parse the !include tag in YAML files to include other YAML files.

    Args:
        loader (yaml.Loader): The YAML loader.
        node (yaml.Node): The YAML node.

    Returns:
        Union[None, Dict]: The included YAML file as a dictionary.

    """
    # loader.name holds the path to the current file being processed
    base_path = os.path.dirname(loader.name)
    included_file = os.path.join(base_path, node.value)
    try:
        with open(included_file, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error("File not found: %s", included_file)
        return None
    except Exception as e:
        logger.error("Error loading file: %s", str(e))
        return None


def read_yml_file(file_path: str) -> dict:
    """ 
    Read a YAML file and return the data as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The data from the YAML file as a dictionary.
    """
    # Add the custom constructor to the yaml loader
    yaml.add_constructor('!include', _include_yaml)
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_default_config_yml(config_name: str, config_subdir: str = None) -> str:
    """
    Get the default configuration YAML based on the name of the configuration.

    Args:
        config_name (str): The name of the configuration (e.g., "cw_attack").
        config_subdir (str, optional): The sub-directory to search in. Defaults to None.

    Returns:
        str: The path to the configuration YAML file.
    """
    if config_name is None:
        raise ValueError("config_name must be specified and not None!")
    file_paths = []
    start_dir = config_path if config_subdir is None else os.path.join(
        config_path, config_subdir)
    for dirpath, _, files in os.walk(start_dir):
        file_paths.extend([os.path.join(dirpath, f)
                           for f in files if f == config_name])

    if not file_paths:
        raise FileNotFoundError(f"Config file {config_name} not found!")
    return file_paths[0]
