
from dataclasses import dataclass
from typing import Dict, TypedDict
from advsecurenet.attacks.adversarial_attack import AdversarialAttack


class AttackConfigDict(TypedDict):
    config: Dict[str, str]


class AttackWithConfigDict(TypedDict):
    attack: AdversarialAttack
    config: AttackConfigDict


@dataclass
class ATCliConfigType:
    """
    This class is used to store the configuration of the adversarial training defense.
    """
    model: str
    models: list[str]
    attacks: list[AttackWithConfigDict]
    dataset_type: str
    num_classes: int
    dataset_path: str
    optimizer: str
    criterion: str
    epochs: int
    batch_size: int
    adv_coeff: float
    verbose: bool
    learning_rate: float
    momentum: float
    weight_decay: float
    scheduler: str
    scheduler_step_size: int
    scheduler_gamma: float
    num_workers: int
    device: str
    save_model: bool
    save_model_path: str
    save_model_name: str
    save_checkpoint: bool
    save_checkpoint_path: str
    save_checkpoint_name: str
    checkpoint_interval: int
    load_checkpoint: bool
    load_checkpoint_path: str
    use_ddp: bool
    gpu_ids: list[int]
    pin_memory: bool
