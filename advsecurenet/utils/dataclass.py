from dataclasses import dataclass, fields, is_dataclass
from typing import (Any, Dict, Generic, List, Optional, Type, TypeVar, Union,
                    get_args, get_origin)

from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig

# This is needed to support recursive dataclass instantiation
T = TypeVar('T')


def flatten_dataclass(instance: object) -> dict:
    """ 
    Recursively flatten dataclass instances into a single dictionary. Recursion is used to flatten nested dataclasses.

    Args:
        instance (object): The dataclass instance to flatten.

    Returns:
        dict: The flattened dataclass instance.
    """
    if not is_dataclass(instance):
        return instance

    result = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        if is_dataclass(value):
            result[field.name] = flatten_dataclass(value)
        else:
            result[field.name] = value
    return result


def filter_for_dataclass(data: Union[dict, object], dataclass_type: type, convert: Optional[bool] = False) -> Union[dict, object]:
    """
    Filter a dictionary to only include keys that are valid fields of the given dataclass type. 

    Args:
        data (Union[dict, dataclass]): The data to filter. If a dataclass instance is provided, it will be flattened first.
        dataclass_type (type): The dataclass type to filter for.
        convert (Optional[bool]): Whether to convert the filtered data back to a dataclass instance. Default is False.

    Returns:
        dict or object: The filtered data. If the convert flag is set to True, the filtered data will be converted to a dataclass instance.
    """
    if is_dataclass(data):
        data = flatten_dataclass(data)
    valid_keys = {field.name for field in fields(dataclass_type)}
    filtered_data = {key: value for key,
                     value in data.items() if key in valid_keys}
    if convert:
        return recursive_dataclass_instantiation(dataclass_type, filtered_data)
    return filtered_data


def recursive_dataclass_instantiation(cls: Type[T], data: dict) -> T:
    """ 
    Recursively instantiate a dataclass from a dictionary. Recursion is used to instantiate nested dataclasses.

    Args:
        cls (Type[T]): The dataclass type to instantiate.
        data (dict): The dictionary to instantiate the dataclass from.

    Returns:
        T: The instantiated dataclass.
    """

    if not is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in fields(cls)}
    new_data = {}

    for key, value in data.items():
        if key not in field_types:
            continue
        field_type = field_types[key]
        origin = get_origin(field_type)
        args = get_args(field_type)
        try:
            if origin is Union and type(None) in args:
                actual_type = next(
                    arg for arg in args if arg is not type(None))
                if is_dataclass(actual_type) and isinstance(value, dict):
                    new_data[key] = recursive_dataclass_instantiation(
                        actual_type, value)
                else:
                    new_data[key] = value
            elif is_dataclass(field_type) and isinstance(value, dict):
                new_data[key] = recursive_dataclass_instantiation(
                    field_type, value)
            elif origin is list and is_dataclass(args[0]) and isinstance(value, list):
                new_data[key] = [recursive_dataclass_instantiation(
                    args[0], item) for item in value]
            # This is for generic types
            elif origin and args and is_dataclass(args[0]) and isinstance(value, dict):
                # Dynamically determine the generic type and assign the key
                additional_fields = {
                    field.name: arg for field, arg in zip(fields(origin), args)}

                # Inject additional fields into the value
                for additional_key, additional_type in additional_fields.items():
                    if additional_key in value and isinstance(value[additional_key], dict):
                        value[additional_key] = recursive_dataclass_instantiation(
                            additional_type, value[additional_key])
                    else:
                        value[additional_key] = value.get(additional_key)

                new_data[key] = recursive_dataclass_instantiation(
                    origin, value)
            elif is_dataclass(origin):
                new_data[key] = recursive_dataclass_instantiation(
                    origin, value)
            else:
                new_data[key] = value
        except Exception as e:
            new_data[key] = value

    return cls(**new_data)


def merge_dataclasses(*dataclasses: object) -> object:
    """
    Merge two dataclasses into a single dataclass. The fields
    of the second dataclass will overwrite the fields of the first dataclass.

    Args:
        dataclasses (object): The dataclasses to merge.
    Returns:
        object: The merged dataclass.

    """

    if len(dataclasses) == 1:
        return dataclasses[0]

    flattened_data = {}
    for current_dataclass in dataclasses:
        if not is_dataclass(current_dataclass):
            continue
        flattened_data.update(flatten_dataclass(current_dataclass))

    return recursive_dataclass_instantiation(type(dataclasses[0]), flattened_data)
