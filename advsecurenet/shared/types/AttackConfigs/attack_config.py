from abc import ABC

"""
AttackConfig is an abstract class that defines the interface for all attack configurations. It's used as a type hint for the config parameter of the attack's constructor. All attack configurations must inherit from this class.
"""
class AttackConfig(ABC):
    pass