"""
This example python script shows how to use the DDPTrainer class to train a model using DistributedDataParallel in a multi-GPU setting.

Make sure to change the port number if the default port 12355 is not free on your machine. And also change the gpu_ids list to the GPUs you want to use.
To use the script, run the following command:
    `python ddp_at.py`
"""

import os
import advsecurenet.shared.types.configs.attack_configs as AttackConfigs
from torch.utils.data.distributed import DistributedSampler
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.datasets import DatasetFactory
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types import DatasetType
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.defenses.multi_adversarial_training import MultiGPUAdversarialTraining
from advsecurenet.attacks.fgsm import FGSM
from advsecurenet.utils.ddp_trainer import DDPTrainer


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
        distributed_mode=True)

    # Create and run the trainer
    trainer = MultiGPUAdversarialTraining(
        adv_training_config, rank, world_size)
    print(f"Training on rank {rank}")
    trainer.train()


def run_training(world_size: int, save_every: int, total_epochs: int, batch_size: int, gpu_ids: list[int]) -> None:
    """ 
    Run the training function using DDP.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    # Create and run the DDP trainer
    ddp_trainer = DDPTrainer(
        main_training_function,  # Training function
        world_size,              # World size
        save_every=save_every,
        total_epochs=total_epochs,
        batch_size=batch_size)

    ddp_trainer.run()

    print("Training complete!")


if __name__ == "__main__":
    gpu_ids = [1, 3]
    world_size = len(gpu_ids)
    save_every = 1
    total_epochs = 1
    batch_size = 64 * world_size

    run_training(world_size, save_every, total_epochs, batch_size, gpu_ids)
