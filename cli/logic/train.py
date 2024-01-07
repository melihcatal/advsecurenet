import click
from cli.types.training import TrainingCliConfigType
from cli.utils.config import load_configuration
from cli.utils.trainer import CLITrainer

from advsecurenet.shared.types.configs import ConfigType
from advsecurenet.utils.config_utils import get_default_config_yml


def cli_train(config: str, **kwargs):
    """Train model."""

    if not config:
        click.echo(
            "No configuration file provided for training! Using default configuration...")
        config = get_default_config_yml("train_config.yml", "cli")

    config_data = load_configuration(
        config_type=ConfigType.TRAIN, config_file=config, **kwargs)
    config_data = TrainingCliConfigType(**config_data)
    trainer = CLITrainer(config_data)
    trainer.train()
