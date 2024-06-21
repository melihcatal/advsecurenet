from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from advsecurenet.defenses import AdversarialTraining
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import \
    AdversarialTrainingConfig
from advsecurenet.trainer.ddp_trainer import DDPTrainer


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

    def _get_train_loader(self, epoch: int):
        sampler = self.config.train_loader.sampler
        assert isinstance(
            sampler, DistributedSampler), "Sampler must be a DistributedSampler"
        sampler.set_epoch(epoch)

        if self._rank == 0:
            return tqdm(self.config.train_loader,
                        desc="Adversarial Training",
                        leave=False,
                        position=1,
                        unit="batch",
                        colour="blue")
        else:
            return self.config.train_loader

    def _get_loss_divisor(self):
        return len(self.config.train_loader) * self._world_size
