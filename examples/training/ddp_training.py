"""
This example script shows how to use the DDPTrainingCoordinator class to train a model using DistributedDataParallel in a multi-GPU setting.
"""

import os
import time
from typing import Optional

import torch
from torch.utils.data.distributed import DistributedSampler

import advsecurenet.shared.types.configs.attack_configs as AttackConfigs
from advsecurenet.attacks.fgsm import FGSM
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.datasets import DatasetFactory
from advsecurenet.defenses.ddp_adversarial_training import \
    DDPAdversarialTraining
from advsecurenet.evaluation.tester import Tester
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types import DatasetType
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.trainer.ddp_trainer import DDPTrainer
from advsecurenet.trainer.ddp_training_coordinator import \
    DDPTrainingCoordinator


def main_training_function(rank, world_size, save_every, total_epochs, batch_size):
    """
    This is the main training function that is called by each process. It adversarially trains a model using FGSM attack in a multi-GPU setting.
    """
    # Load the model and dataset
    mnist_model = ModelFactory.create_model("resnet18", num_classes=10)
    dataset = DatasetFactory.create_dataset(DatasetType.CIFAR10)
    train_data = dataset.load_dataset(train=True)

    # Initialize the data loader with DistributedSampler
    train_loader = DataLoaderFactory.create_dataloader(
        train_data, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=DistributedSampler(train_data), num_workers=1)
    # Initialize the train config
    train_config = TrainConfig(
        model=mnist_model,
        train_loader=train_loader,
        epochs=total_epochs,
        save_final_model=True,
    )
    # Create and run the trainer
    trainer = DDPTrainer(train_config, rank, world_size)
    trainer.train()


def run_training(world_size: int, save_every: int, total_epochs: int, batch_size: int, gpu_ids: Optional[list[int]] = None) -> None:
    """ 
    Run the training function using DDP.

    Args:
        world_size (int): The number of processes to spawn.
        save_every (int): The number of epochs after which the model is saved.
        total_epochs (int): The total number of epochs to train.
        batch_size (int): The batch size per GPU.
        gpu_ids (Optional[list[int]], optional): The list of GPU IDs to use. Defaults to None. If None, all available GPUs are used.
    """
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    # Create and run the DDP trainer
    ddp_trainer = DDPTrainingCoordinator(
        main_training_function, world_size, save_every, total_epochs, batch_size
    )
    ddp_trainer.run()
    print("Training complete!")


def testing():
    file = "resnet18_cifar10.pth"  # Change this to the path of the saved model
    # Load the model and dataset
    model = ModelFactory.create_model("resnet18", num_classes=10)
    model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    model.eval()
    dataset = DatasetFactory.create_dataset(DatasetType.CIFAR10)
    test_data = dataset.load_dataset(train=False)
    test_loader = DataLoaderFactory.create_dataloader(
        test_data, batch_size=64, shuffle=False)
    # Test the model
    device = torch.device("cuda:1")
    tester = Tester(model, test_loader, device=device)
    tester.test()


if __name__ == "__main__":
    gpu_ids = [2, 5]
    world_size = len(gpu_ids)
    save_every = 1
    total_epochs = 5
    batch_size = 64 * world_size
    run_training(world_size, save_every, total_epochs, batch_size, gpu_ids)
    testing()
