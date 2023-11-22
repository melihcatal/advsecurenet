
from dataclasses import dataclass
from typing import Dict, TypedDict
from advsecurenet.attacks.adversarial_attack import AdversarialAttack


class AttackConfigDict(TypedDict):
    config: Dict[str, str]


class AttackWithConfigDict(TypedDict):
    attack: AdversarialAttack
    config: AttackConfigDict


class ModelConfigDict(TypedDict):
    is_custom: bool
    pretrained: bool
    weights_path: str


class ModelWithConfigDict(TypedDict):
    model: str
    config: ModelConfigDict


@dataclass
class ATCliConfigType:
    """
    This class is used as a type hint for the Adversarial Training CLI configuration.
    """
    model: str
    models: list[ModelWithConfigDict]
    attacks: list[AttackWithConfigDict]
    dataset_type: str
    num_classes: int
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
    num_workers_train: int
    num_workers_test: int
    shuffle_train: bool
    shuffle_test: bool
    drop_last_train: bool
    drop_last_test: bool
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
    train_dataset_path: str
    test_dataset_path: str
    num_samples_train: int
    num_samples_test: int
