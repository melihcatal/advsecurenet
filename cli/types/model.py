from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelCliConfigType:
    """
    This dataclass is used to store the configuration of the model CLI.
    """
    model_name: str
    num_classes: int
    num_input_channels: int
    add_norm_layer: bool
    norm_mean: Optional[List[float]]
    norm_std: Optional[List[float]]
