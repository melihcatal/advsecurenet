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
        DDPTrainer.__init__(self, config=config, rank=rank,
                            world_size=world_size)
        AdversarialTraining.__init__(self, config=config)
