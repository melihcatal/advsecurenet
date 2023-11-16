"""
This example python script shows how to use the DDPTrainingCoordinator class to train a model using DistributedDataParallel in a multi-GPU setting.

Make sure to change the port number if the default port 12355 is not free on your machine. And also change the gpu_ids list to the GPUs you want to use.
To use the script, run the following command:
    `python ddp_at.py`
"""

import os
import torch
import advsecurenet.shared.types.configs.attack_configs as AttackConfigs
from torch.utils.data.distributed import DistributedSampler
from typing import Optional
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.datasets import DatasetFactory
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types import DatasetType
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.defenses.ddp_adversarial_training import DDPAdversarialTraining
from advsecurenet.attacks.fgsm import FGSM
from advsecurenet.utils.ddp_training_coordinator import DDPTrainingCoordinator
from advsecurenet.utils.tester import Tester


def main_training_function(rank, world_size, save_every, total_epochs, batch_size):
    """
    This is the main training function that is called by each process. It adversarially trains a model using FGSM attack in a multi-GPU setting.
    """
    print(f"Running basic DDP example on rank {rank}.")

    # Load the model and dataset
    mnist_model = ModelFactory.get_model("resnet18", num_classes=10)
    dataset = DatasetFactory.load_dataset(DatasetType.CIFAR10)
    train_data = dataset.load_dataset(train=True)

    # Initialize the data loader with DistributedSampler
    train_loader = DataLoaderFactory.get_dataloader(
        train_data, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=DistributedSampler(train_data))

    # Initialize the attack and training configurations
    fgsm_config = AttackConfigs.FgsmAttackConfig(distributed_mode=True)
    fgsm = FGSM(fgsm_config)
    adv_training_config = AdversarialTrainingConfig(
        model=mnist_model,
        models=[mnist_model],
        attacks=[fgsm],
        train_loader=train_loader,
        epochs=total_epochs,
        use_ddp=True)

    # Create and run the trainer
    trainer = DDPAdversarialTraining(
        adv_training_config, rank, world_size)
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
    model = ModelFactory.get_model("resnet18", num_classes=10)
    model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    model.eval()
    dataset = DatasetFactory.load_dataset(DatasetType.CIFAR10)
    test_data = dataset.load_dataset(train=False)
    test_loader = DataLoaderFactory.get_dataloader(
        test_data, batch_size=64, shuffle=False, pin_memory=True)
    # Test the model
    device = torch.device("cuda:1")
    tester = Tester(model, test_loader, device=device)
    tester.test()


if __name__ == "__main__":
    gpu_ids = [2, 7]
    world_size = len(gpu_ids)
    save_every = 1
    total_epochs = 1
    batch_size = 64 * world_size
    run_training(world_size, save_every, total_epochs, batch_size, gpu_ids)
    testing()
