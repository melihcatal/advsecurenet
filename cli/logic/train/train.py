import logging

from advsecurenet.shared.types.configs import ConfigType
from cli.logic.train.trainer import CLITrainer
from cli.shared.types.train import TrainingCliConfigType
from cli.shared.utils.config import load_and_instantiate_config

logger = logging.getLogger(__name__)


def cli_train(config: str, **kwargs) -> None:
    """Main function for training a model using the CLI.

    Args:
        config (str): The path to the configuration file.
        **kwargs: Additional keyword arguments.
    """
    config_data: TrainingCliConfigType = load_and_instantiate_config(
        config, "train_config.yml", ConfigType.TRAIN, TrainingCliConfigType, **kwargs)
    logger.info("Loaded training configuration: %s", config_data)
    try:
        trainer = CLITrainer(config_data)
        trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error("Failed to train model: %s", e)
        raise e
