from enum import Enum
from torch import device


class DeviceType(Enum):
    CUDA = device("cuda")
    CUDA_0 = device("cuda:0")
    CPU = device("cpu")
    MPS = device("mps")
