from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessStep:
    """
    This dataclass is used to store the configuration of a preprocessing step.
    """
    name: str
    params: Optional[dict] = None


@dataclass
class PreprocessConfig:
    """
    This dataclass is used to store the configuration of the preprocessing pipeline.
    """
    steps: Optional[list[PreprocessStep]] = None
