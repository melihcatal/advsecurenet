from dataclasses import dataclass
from torch.optim import Optimizer
from torch import nn
from advsecurenet.models.base_model import BaseModel
from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.device import DeviceType
from advsecurenet.shared.types.configs.defense_configs import DefenseConfig


@dataclass
class AdversarialTrainingConfig(DefenseConfig):
    """
    This class is used to store the configuration of the adversarial training defense.

    Attributes
    ----------
    target_model: BaseModel
        The model that will be trained.
    models: list[BaseModel]
        A list of models that will be used to generate adversarial examples.
    attacks: list[AdversarialAttack]
        A list of attacks that will be used to generate adversarial examples.
    train_dataloader: DataLoaderFactory
        A dataloader that will be used to train the model.
    optimizer: Optimizer
        This is an optimizer such as torch.optim.Adam().
    criterion: nn.Module
        This is a loss function such as nn.CrossEntropyLoss().
    epochs: int
        The number of epochs to train the model.
    adv_coeff: float
        The coefficient that will be used to combine the clean and adversarial examples.
    device: DeviceType
        The device that will be used to train the model.
    verbose: bool
        Whether to print progress or not. 
    """
    target_model: BaseModel
    models: list[BaseModel]
    attacks: list[AdversarialAttack]
    train_dataloader: DataLoaderFactory
    optimizer: Optimizer
    criterion: nn.Module
    epochs: int = 5
    adv_coeff: float = 0.5
    device: DeviceType = DeviceType.CPU
    verbose: bool = True
