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
    model_weights_path: Optional[str]


@dataclass
class ModelCliConfigType:
    """
    This dataclass is used to store the configuration of the model CLI.
    """
    model_name: str
    num_input_channels: int
    num_classes: int
    pretrained: bool
    weights: Optional[str]
    is_external: bool
    path_configs: ModelPathConfig
    norm_config: ModelNormConfig
    random_seed: Optional[int]
