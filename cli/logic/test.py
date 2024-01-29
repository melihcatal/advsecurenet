import click
from cli.utils.config import load_configuration
from cli.utils.model import \
    cli_test as \
    test  # cli_test here is the utility function. Don't confuse it with the cli_test function below.

from advsecurenet.shared.types.configs import ConfigType, TestConfig
from advsecurenet.utils.config_utils import get_default_config_yml


def cli_test(config: str, **kwargs):
    """Test model."""
    if not config:
        click.echo(
            "No configuration file provided for evaluation! Using default configuration...")
        config = get_default_config_yml("test_config.yml", "cli")

    config_data: TestConfig = load_configuration(
        config_type=ConfigType.TEST, config_file=config, **kwargs)
    config_data = TestConfig(**config_data)
    test(config_data)
