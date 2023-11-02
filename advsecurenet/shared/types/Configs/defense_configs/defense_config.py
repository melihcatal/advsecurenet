from abc import ABC 

"""
DefenseConfig is an abstract class that defines the interface for all defense configurations. It's used as a type hint for the config parameter of the defense's constructor. All defense configurations must inherit from this class.
"""

class DefenseConfig(ABC):
    pass