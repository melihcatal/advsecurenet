from dataclasses import dataclass
from typing import Optional


@dataclass
class AttackProcedureCliConfigType:
    """
    This dataclass is used to store the configuration of the attack procedure CLI.
    """
    verbose: Optional[bool] = True
    save_result_images: Optional[bool] = False
    result_images_dir: Optional[str] = None
    result_images_prefix: Optional[str] = None
