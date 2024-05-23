from dataclasses import fields, is_dataclass
from typing import Optional, Union, get_args, get_origin


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


def recursive_dataclass_instantiation(cls: type, data: dict) -> type:
    """
    Recursively instantiate a dataclass with nested dataclasses from a dictionary. 
    A dictionary may contain nested dictionaries that represent nested dataclasses. Recursion is used to instantiate the nested dataclasses.

    Args:   
        cls (type): The dataclass type to instantiate.
        data (dict): The data to instantiate the dataclass with.

    Returns:
        The instantiated dataclass.
    """
    field_types = {f.name: f.type for f in fields(cls)}
    new_data = {}
    for key, value in data.items():
        # Check if the field type is an Optional or directly a dataclass
        field_type = field_types[key]
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union and type(None) in args:
            # If the type is Optional[SomeType], get the actual SomeType
            actual_type = next(arg for arg in args if arg is not type(None))
            if is_dataclass(actual_type) and isinstance(value, dict):
                new_data[key] = recursive_dataclass_instantiation(
                    actual_type, value)
            else:
                new_data[key] = value
        elif is_dataclass(field_type) and isinstance(value, dict):
            new_data[key] = recursive_dataclass_instantiation(
                field_type, value)
        else:
            new_data[key] = value
    return cls(**new_data)


def merge_dataclasses(dataclass1: object, dataclass2: object) -> object:
    """
    Merge two dataclasses into a single dataclass. The fields
    of the second dataclass will overwrite the fields of the first dataclass.

    Args:
        dataclass1 (object): The first dataclass.
        dataclass2 (object): The second dataclass.

    Returns:
        object: The merged dataclass.

    """
    if not is_dataclass(dataclass1) or not is_dataclass(dataclass2):
        return dataclass1

    data1 = flatten_dataclass(dataclass1)
    data2 = flatten_dataclass(dataclass2)
    data1.update(data2)
    return recursive_dataclass_instantiation(type(dataclass1), data1)
