
from advsecurenet.shared.types.configs import ConfigType
from cli.types.testing import TestingCliConfigType
from cli.utils.config import load_and_instantiate_config
from cli.utils.tester import CLITester


def cli_test(config: str, **kwargs) -> None:
    """Main function for testing a model using the CLI.

    Args:
        config (str): The path to the configuration file.
        **kwargs: Additional keyword arguments.
    """
    config_data = load_and_instantiate_config(
        config, "test_config.yml", ConfigType.TEST, TestingCliConfigType, **kwargs)

    tester = CLITester(config_data)
    tester.test()
