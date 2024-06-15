from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class BaseModelConfig:
    """
    Base configuration class for different model configurations.
    """
    model_name: str
    num_classes: Optional[int] = 1000
    num_input_channels: Optional[int] = 3
    pretrained: Optional[bool] = False


@dataclass
class StandardModelConfig(BaseModelConfig):
    """
    Configuration for a standard model.
    """
    weights: Optional[str] = "IMAGENET1K_V1"


@dataclass
class CustomModelConfig(BaseModelConfig):
    """
    Configuration for a custom model.
    """
    custom_models_path: Optional[str] = "CustomModels"


@dataclass
class ExternalModelConfig(BaseModelConfig):
    """
    Configuration for an external model.
    """
    model_arch_path: Optional[str] = None
    model_weights_path: Optional[str] = None


@dataclass
class CreateModelConfig(StandardModelConfig, CustomModelConfig, ExternalModelConfig):
    """
    Config parameters for creating a model in the model factory.
    """
    is_external: bool = False
    random_seed: Optional[int] = None
