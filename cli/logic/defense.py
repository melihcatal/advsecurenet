import click

from advsecurenet.shared.types.configs import ConfigType
from cli.types.adversarial_training import ATCliConfigType
from cli.utils.adversarial_training_cli import AdversarialTrainingCLI
from cli.utils.config import load_configuration


def cli_adversarial_training(config: str, **kwargs) -> None:
    """
    Logic function to execute adversarial training.
    """
    if not config:
        raise click.ClickException(
            "No configuration file provided for adversarial training! Use the 'config-default' command to generate a default configuration file.")

    config_data = load_configuration(
        config_type=ConfigType.DEFENSE, config_file=config, **kwargs)
    config_data = ATCliConfigType(**config_data)
    adversarial_training = AdversarialTrainingCLI(config_data)
    adversarial_training.train()
