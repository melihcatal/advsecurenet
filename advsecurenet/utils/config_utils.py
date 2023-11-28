import os
import re
from dataclasses import asdict
from typing import Type, TypeVar

import pkg_resources
import yaml
from ruamel.yaml import YAML

T = TypeVar('T')
config_path = pkg_resources.resource_filename("advsecurenet", "configs")


def load_config_from_yaml(yaml_path: str, config_class: Type[T]) -> T:
    """
    Load a YAML file into a dataclass.

    Args:
        yaml_path (str): The path to the YAML file.
        config_class (Type[T]): The dataclass type to load the YAML into.

    Returns:
        T: The dataclass with the YAML data loaded into it.


    Examples:

        >>> from dataclasses import dataclass
        >>> from typing import List
        >>> from advsecurenet.utils.config_loader import load_config_from_yaml 
        >>>
        >>> @dataclass
        >>> class Config:
        >>>     a: int = 0
        >>>     b: str = "test"
        >>>     c: List[int] = None
        >>>
        >>> # config.yml
        >>> # a: 1
        >>> # b: test
        >>> # c: [1, 2, 3]
        >>> config = load_config_from_yaml("config.yaml", Config)
        >>> print(config)
        Config(a=1, b='test', c=[1, 2, 3])
    """

    # Start with dataclass defaults
    config_data = asdict(config_class())

    # Load the YAML and override defaults
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        for key, value in yaml_data.items():
            if key in config_data:  # Ensure that the YAML key corresponds to a valid dataclass field
                config_data[key] = value

    return config_class(**config_data)


def save_config_to_yaml(config: T, yaml_path: str) -> None:
    """
    Save a dataclass to a YAML file.

    Args:
        config (T): The dataclass to save.
        yaml_path (str): The path to the YAML file.

    """

    with open(yaml_path, 'w') as file:
        yaml.dump(asdict(config), file, default_flow_style=False)


def load_default_config(config_class: Type[T]) -> T:
    """
    Load the default configuration for a dataclass.

    Args:
        config_class (Type[T]): The dataclass type to load the default configuration for.

    Returns:
        T: The dataclass with the default configuration loaded into it.

    """
    file_name = get_config_file_name(config_class)
    path = pkg_resources.resource_filename("advsecurenet", "configs")
    default_yaml_path = os.path.join(path, file_name)
    return load_config_from_yaml(default_yaml_path, config_class)


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

    if not config_name.endswith("_config.yml"):
        config_name = config_name + "_config.yml"

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
        yaml = YAML()
        print(f"Saving default config to {output_path}")
        # if the file does not exist, create it otherwise overwrite it
        with open(output_path, 'w+') as file:
            yaml.dump(default_config, file)

    return default_config


def get_config_file_name(config_class: Type[T]) -> str:
    """
    Split the class name into words it's a camel case string like "LotsAttackConfig"

    Args:
        config_class (Type[T]): The dataclass type to get the config file name for.

    Returns:
        str: The config file name.

    Examples:
        >>> from advsecurenet.shared.types.configs import LotsAttackConfig
        >>> from advsecurenet.utils.config_utils import get_config_file_name
        >>> get_config_file_name(LotsAttackConfig)
        'lots_attack_config.yml'
    """
    words = re.findall('[A-Z][^A-Z]*', config_class.__name__)
    # join the words with underscores and make it lowercase
    return '_'.join(words).lower() + '.yml'


def get_available_configs() -> list:
    """
    Get a list of all available configuration files, including those in subdirectories.

    Returns:
        list: A list of all available configuration files.

    Examples:
        >>> from advsecurenet.utils.config_utils import get_available_configs
        >>> get_available_configs()
        ['lots_attack_config.yml', 'cw_attack_config.yml', ...]

    """
    cli_configs_dir = os.path.join(config_path, "cli")
    config_files = []
    for root, dirs, files in os.walk(cli_configs_dir):
        for file in files:
            if file.endswith("_config.yml"):
                # add the name of the file to the list
                config_files.append(file)

    return config_files


def read_yml_file(yml_path: str) -> dict:
    """
    Read a YAML file and return it as a dictionary.

    Args:
        yml_path (str): The path to the YAML file.

    Returns:
        dict: The YAML file as a dictionary.

    """
    yaml = YAML()
    with open(yml_path, 'r') as file:
        return yaml.load(file)


def get_default_config_yml(config_name: str, config_subdir: str = None):
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


def override_with_cli_args(config_data, **cli_args):
    """
    Override values in config_data with those from cli_args if provided.

    Args:
    - config_data (dict): Dictionary of configuration from a file or default values.
    - **cli_args: Arguments provided from the CLI.

    Returns:
    - Dictionary with overridden values.
    """
    for key, value in cli_args.items():
        if value is not None:  # CLI argument was provided
            config_data[key] = value
        elif key not in config_data:
            # Add keys that might not exist in config_data without a value
            config_data[key] = None

    return config_data
