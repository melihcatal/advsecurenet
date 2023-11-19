import torch
from dataclasses import dataclass, field
from typing import List, Union, Optional, Dict, TypedDict
from torch.optim import Optimizer
from torch import nn
from torch.utils.data import DataLoader
from advsecurenet.models.base_model import BaseModel
from advsecurenet.attacks.adversarial_attack import AdversarialAttack
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
    target_model : BaseModel
        The model that will be trained.
    models : List[BaseModel]
        A list of models that will be used to generate adversarial examples.
    attacks : List[AdversarialAttack]
        A list of attacks that will be used to generate adversarial examples along with their configurations.
    train_dataloader : DataLoader
        A dataloader that will be used to train the model.
    optimizer : Optimizer
        This is an optimizer such as torch.optim.Adam().
    criterion : nn.Module
        This is a loss function such as nn.CrossEntropyLoss().
    epochs : int
        The number of epochs to train the model.
    learning_rate : float
        The learning rate for the optimizer.
    device : torch.device
        The device that will be used to train the model.
    save_checkpoint : bool
        Whether to save model checkpoints.
    checkpoint_path : Optional[str]
        Path where checkpoints will be saved.
    checkpoint_interval : int
        Interval between saving checkpoints.
    load_checkpoint : bool
        Whether to load a model checkpoint.
    load_checkpoint_path : Optional[str]
        Path from where the checkpoint will be loaded.
    verbose : bool
        Whether to print progress or not.
    adv_coeff : float
        The coefficient that will be used to combine the clean and adversarial examples.
    use_ddp : bool
        Whether to use DistributedDataParallel or not.
    gpu_ids : Optional[List[int]]
        A list of GPU IDs to use. If None, all available GPUs are used.
    pin_memory : bool
        Whether to pin memory or not.
    lots_target_images : Optional[torch.Tensor]
        A tensor containing the target images for LOTS attack. This is only used if LOTS attack is used.
    lots_target_labels : Optional[torch.Tensor]
        A tensor containing the target labels for LOTS attack. This is only used if LOTS attack is used.
    """
    model: BaseModel  # the target model
    models: List[BaseModel]
    attacks: List[AdversarialAttack]
    train_loader: DataLoader
    lots_target_images: Optional[torch.Tensor] = None
    lots_target_labels: Optional[torch.Tensor] = None
    lots_data_loader: Optional[DataLoader] = None
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
