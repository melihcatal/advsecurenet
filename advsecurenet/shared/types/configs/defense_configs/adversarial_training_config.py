from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypedDict, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from advsecurenet.attacks.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.defense_configs import DefenseConfig
from advsecurenet.shared.types.configs.train_config import TrainConfig

# class AttackConfigDict(TypedDict):
#     config: Dict[str, str]


# class AttackWithConfigDict(TypedDict):
#     attack: AdversarialAttack
#     config: AttackConfigDict


@dataclass(kw_only=True)
class AdversarialTrainingConfig(TrainConfig, DefenseConfig):
    """
    This class is used to store the configuration of the adversarial training defense.

    Args:
        model (BaseModel): The target model.
        models (List[BaseModel]): The list of models to be trained.
        attacks (List[AdversarialAttack]): The list of attacks to be used.
        train_loader (DataLoader): The training data loader.
        optimizer (Union[str, Optimizer], optional): The optimizer to be used. Defaults to "adam".
        criterion (Union[str, nn.Module], optional): The loss function to be used. Defaults to "cross_entropy".
        epochs (int, optional): The number of epochs to be used. Defaults to 5.
        learning_rate (float, optional): The learning rate to be used. Defaults to 0.001.
        save_checkpoint (bool, optional): Whether to save the checkpoints. Defaults to False.
        save_checkpoint_path (Optional[str], optional): The path to save the checkpoints. Defaults to None.
        save_checkpoint_name (Optional[str], optional): The name of the checkpoint to be saved. Defaults to None.
        checkpoint_interval (int, optional): The interval between checkpoints. Defaults to 1.
        load_checkpoint (bool, optional): Whether to load the checkpoints. Defaults to False.
        load_checkpoint_path (Optional[str], optional): The path to load the checkpoints. Defaults to None.
        verbose (bool, optional): Whether to print the logs. Defaults to True.
        adv_coeff (float, optional): The coefficient for the adversarial loss. Defaults to 0.5.
        device (torch.device, optional): The device to be used. Defaults to torch.device("cpu").
        use_ddp (bool, optional): Whether to use DistributedDataParallel. Defaults to False.
        gpu_ids (Optional[List[int]], optional): The list of GPU IDs to be used. Defaults to None.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.

    """
    model: BaseModel  # the target model
    models: List[BaseModel]
    attacks: List[AdversarialAttack]
    train_loader: DataLoader
    optimizer: Union[str, Optimizer] = "adam"
    criterion: Union[str, nn.Module] = "cross_entropy"
    epochs: int = 5
    learning_rate: float = 0.001
    save_checkpoint: bool = False
    save_checkpoint_path: Optional[str] = None
    save_checkpoint_name: Optional[str] = None
    checkpoint_interval: int = 1
    load_checkpoint: bool = False
    load_checkpoint_path: Optional[str] = None
    verbose: bool = True
    adv_coeff: float = 0.5
    device: torch.device = torch.device("cpu")
    use_ddp: bool = False
    gpu_ids: Optional[List[int]] = None
    pin_memory: bool = False

    def __setattr__(self, prop, value):
        if prop == "device":
            value = self._check_device(value)
        super().__setattr__(prop, value)

    @staticmethod
    def _check_device(device: Union[str, torch.device]):
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except:
                device = torch.device("cpu")
        return device
