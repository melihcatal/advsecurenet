from dataclasses import fields, is_dataclass
from typing import Optional, Type, TypeVar, Union, get_args, get_origin

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
    new_data = {key: process_field(
        field_types[key], value) for key, value in data.items() if key in field_types}

    return cls(**new_data)


def process_field(field_type: Type, value):
    origin = get_origin(field_type)
    args = get_args(field_type)

    if is_optional_type(field_type):
        return process_optional_field(args, value)
    elif is_dataclass(field_type) and isinstance(value, dict):
        return recursive_dataclass_instantiation(field_type, value)
    elif is_list_of_dataclass(field_type, value):
        return [recursive_dataclass_instantiation(args[0], item) for item in value]
    elif origin and args and is_dataclass(args[0]) and isinstance(value, dict):
        return process_generic_type(origin, args, value)
    elif is_dataclass(origin):
        return recursive_dataclass_instantiation(origin, value)
    else:
        return value


def is_optional_type(field_type: Type) -> bool:
    origin = get_origin(field_type)
    args = get_args(field_type)
    return origin is Union and type(None) in args


def process_optional_field(args, value):
    actual_type = next(arg for arg in args if arg is not type(None))
    if is_dataclass(actual_type) and isinstance(value, dict):
        return recursive_dataclass_instantiation(actual_type, value)
    return value


def is_list_of_dataclass(field_type: Type, value) -> bool:
    origin = get_origin(field_type)
    args = get_args(field_type)
    return origin is list and is_dataclass(args[0]) and isinstance(value, list)


def process_generic_type(origin, args, value):
    additional_fields = {field.name: arg for field,
                         arg in zip(fields(origin), args)}
    for additional_key, additional_type in additional_fields.items():
        if additional_key in value and isinstance(value[additional_key], dict):
            value[additional_key] = recursive_dataclass_instantiation(
                additional_type, value[additional_key])
        else:
            value[additional_key] = value.get(additional_key)
    return recursive_dataclass_instantiation(origin, value)


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
