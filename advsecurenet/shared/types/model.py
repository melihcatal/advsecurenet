from enum import Enum


class ModelType(Enum):
    """
    Specifies the type of the model.
    Supported types are:
    - STANDARD: The model is a standard model from torchvision.models.
    - CUSTOM: The model is a custom model provided by the package that is not present in torchvision.models. These models can be used for research purposes.
    - EXTERNAL: The model is an external model that is not provided by the package. These models are loaded from external Python files.
    """
    STANDARD = "standard"
    CUSTOM = "custom"
    EXTERNAL = "external"
