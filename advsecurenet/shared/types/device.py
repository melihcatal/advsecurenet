from enum import Enum
from torch import device


class DeviceType(Enum):
    CUDA = device("cuda")
    CUDA_0 = device("cuda:0")
    CPU = device("cpu")
    MPS = device("mps")
    
    @classmethod
    def from_string(cls, s):
        for member in cls:
            if member.value == device(s.lower()):
                return member
        raise ValueError(f"{s} is not a valid DeviceType")