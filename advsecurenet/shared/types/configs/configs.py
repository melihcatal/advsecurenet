from enum import Enum

"""
Define the possible configurations
"""
class ConfigType(Enum):
    ATTACK = "attack",
    DEFENSE = "defense",
    TRAIN = "train",
    TEST = "test",
