from dataclasses import dataclass
from advsecurenet.shared.types.device import DeviceType
from enum import Enum

class LotsAttackMode(Enum):
    ITERATIVE = "iterative"
    SINGLE = "single"

@dataclass
class LotsAttackConfig:
    deep_feature_layer : str 
    mode : str = LotsAttackMode.ITERATIVE
    epsilon : float = 0.1
    learning_rate : float = 1./255.
    max_iterations : int = 1000
    verbose : bool = True
    device : DeviceType = DeviceType.CPU




