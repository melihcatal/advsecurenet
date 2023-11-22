from tqdm import tqdm
from typing import Union
from torch.utils.data.distributed import DistributedSampler
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.utils.ddp_trainer import DDPTrainer
from advsecurenet.defenses import AdversarialTraining


class DDPAdversarialTraining(DDPTrainer, AdversarialTraining):
    """
    Adversarial Training class. This class is used to train a model using adversarial training.

    Note:
        This module inherits from both DDPTrainer and AdversarialTraining. The order of inheritance is important because of MRO.
    """

    def __init__(self, config: AdversarialTrainingConfig, rank: int, world_size: int) -> None:
        DDPTrainer.__init__(self, config=config,
                            rank=rank,
                            world_size=world_size)
        AdversarialTraining.__init__(self, config=config)

    def _run_epoch(self, epoch: int) -> None:
        if self.rank == 0:
            print(f"Running epoch {epoch}...")

        total_loss = 0.0
        sampler = self.config.train_loader.sampler
        assert isinstance(
            sampler, DistributedSampler), "Sampler must be a DistributedSampler"
        sampler.set_epoch(epoch)

        for batch_idx, (source, targets) in enumerate(tqdm(self.config.train_loader)):

            # Move data to device
            source = source.to(self.device)
            targets = targets.to(self.device)

            # Generate adversarial examples
            adv_source, adv_targets = self._generate_adversarial_batch(
                source=source,
                targets=targets,
            )

            # Move adversarial examples to device
            adv_source = adv_source.to(self.device)
            adv_targets = adv_targets.to(self.device)

            # Combine clean and adversarial examples
            combined_data, combined_targets = self._combine_clean_and_adversarial_data(
                source=source,
                adv_source=adv_source,
                targets=targets,
                adv_targets=adv_targets
            )

            loss = self._run_batch(combined_data, combined_targets)
            total_loss += loss

        # Compute average loss across all batches and all processes
        total_loss /= len(self.config.train_loader) * self.world_size

        if self.rank == 0:
            print(f"Average loss: {total_loss}")
