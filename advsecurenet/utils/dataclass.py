from dataclasses import fields, is_dataclass


def flatten_dataclass(instance: object) -> dict:
    """ 
    Recursively flatten dataclass instances into a single dictionary. Recursion is used to flatten nested dataclasses.

    Args:
        instance (object): The dataclass instance to flatten.

    Returns:
        dict: The flattened dataclass instance.

    Example:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Data:
        ...     name: str
        ...     age: int
        ...
        >>> data = Data(name='John', age=30)
        >>> flatten_dataclass(data)
        {'name': 'John', 'age': 30}

    """
    if not is_dataclass(instance):
        return instance

    result = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        if is_dataclass(value):
            result.update(flatten_dataclass(value))
        else:
            result[field.name] = value
    return result


def filter_for_dataclass(data: dict,
                         dataclass_type: type) -> dict:
    """
    Filter a dictionary to only include keys that are valid fields of the given dataclass type. 

    Args:
        data (dict): The dictionary to filter.
        dataclass_type (type): The dataclass type to filter for.

    Returns:
        dict: The filtered dictionary.

    Example:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Data:
        ...     name: str
        ...     age: int
        ...
        >>> data = {'name': 'John',
        ...         'age': 30,
        ...
        ...         'invalid_key': 'value'}
        ...
        >>> filter_for_dataclass(data, Data)
        {'name': 'John', 'age': 30}

    """
    valid_keys = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in data.items() if key in valid_keys}


def recursive_dataclass_instantiation(cls: type, data: dict) -> type:
    """
    Recursively instantiate a dataclass with nested dataclasses from a dictionary. 
    A dictionary may contain nested dictionaries that represent nested dataclasses. Recursion is used to instantiate the nested dataclasses.
    Args:   
        cls (type): The dataclass type to instantiate.
        data (dict): The data to instantiate the dataclass with.

    Returns:
        The instantiated dataclass.

    Example:
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Data:
        ...     name: str
        ...     age: int
        ...
        ... @dataclass
        ... class NestedData:
        ...     data: Data
        ...     value: int
        ...
        >>> data = {'data': {'name': 'John', 'age': 30},
        ...         'value': 10}
        ...
        >>> recursive_dataclass_instantiation(NestedData, data)
        NestedData(data=Data(name='John', age=30), value=10)
    """
    field_types = {f.name: f.type for f in fields(cls)}
    new_data = {}
    for key, value in data.items():
        if key in field_types and is_dataclass(field_types[key]) and isinstance(value, dict):
            new_data[key] = recursive_dataclass_instantiation(
                field_types[key], value)
        else:
            new_data[key] = value
    return cls(**new_data)
