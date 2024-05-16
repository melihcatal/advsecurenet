
from advsecurenet.shared.types.configs import ConfigType
from cli.logic.defense.adversarial_training.adversarial_training_cli import \
    ATCLITrainer
from cli.shared.types.defense.adversarial_training import ATCliConfigType
from cli.shared.utils.config import load_and_instantiate_config


def cli_adversarial_training(config: str, **kwargs) -> None:
    """
    Logic function to execute adversarial training.

    Args:
        config (str): Path to the adversarial training configuration yml file.
        kwargs: Additional arguments.
    """

    config_data = load_and_instantiate_config(
        config=config,
        default_config_file="adversarial_training_config.yml",
        config_type=ConfigType.ADVERSARIAL_TRAINING,
        config_class=ATCliConfigType,
        **kwargs
    )
    adversarial_training = ATCLITrainer(config_data)
    adversarial_training.train()
