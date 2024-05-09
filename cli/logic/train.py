

from advsecurenet.shared.types.configs import ConfigType
from cli.types.training import TrainingCliConfigType
from cli.utils.config import load_and_instantiate_config
from cli.utils.trainer import CLITrainer


def cli_train(config: str, **kwargs) -> None:
    """Main function for training a model using the CLI.

    Args:
        config (str): The path to the configuration file.
        **kwargs: Additional keyword arguments.
    """
    config_data: TrainingCliConfigType = load_and_instantiate_config(
        config, "train_config.yml", ConfigType.TRAIN, TrainingCliConfigType, **kwargs)
    trainer = CLITrainer(config_data)
    trainer.train()
