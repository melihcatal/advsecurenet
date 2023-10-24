import yaml
from dataclasses import asdict
from typing import Type, TypeVar

T = TypeVar('T')


def load_config_from_yaml(yaml_path: str, config_class: Type[T]) -> T:
    # Start with dataclass defaults
    config_data = asdict(config_class())

    # Load the YAML and override defaults
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        for key, value in yaml_data.items():
            if key in config_data:  # Ensure that the YAML key corresponds to a valid dataclass field
                config_data[key] = value

    return config_class(**config_data)
