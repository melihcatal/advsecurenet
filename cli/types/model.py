from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelNormConfig:
    """ 
    This dataclass is used to store the configuration of the normalization layer of a model.
    """
    add_norm_layer: bool
    norm_mean: Optional[List[float]]
    norm_std: Optional[List[float]]


@dataclass
class ModelPathConfig:
    """
    This dataclass is used to store the configuration of the paths of a model.
    """
    model_arch_path: Optional[str]
    has_weights: bool
    model_weights_path: Optional[str]


@dataclass
class ModelCliConfigType:
    """
    This dataclass is used to store the configuration of the model CLI.
    """
    model_name: str
    is_external: bool
    path_config: ModelPathConfig
    num_classes: int
    num_input_channels: int
    norm_config: ModelNormConfig
