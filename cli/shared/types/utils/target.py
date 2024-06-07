from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TargetImageCLIConfigType:
    """
    This dataclass is used to store the configuration of the target CLI images.
    """
    target_images_dir: Optional[str] = None


@dataclass
class TargetLabelCLIConfigType:
    """
    This dataclass is used to store the configuration of the target CLI labels.
    """
    target_labels: Optional[List[int]] = None
    target_labels_path: Optional[str] = None
    target_labels_separator: Optional[str] = ","


@dataclass
class TargetCLIConfigType:
    """
    This dataclass is used to store the configuration of the target CLI.
    """
    targeted: Optional[bool] = False
    auto_generate_target: Optional[bool] = False
    target_images_config: Optional[TargetImageCLIConfigType] = None
    target_labels_config: Optional[TargetLabelCLIConfigType] = None
