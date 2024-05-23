
from advsecurenet.shared.types.configs import ConfigType
from cli.logic.evaluation.test.tester import CLITester
from cli.shared.types.evaluation.testing import TestingCliConfigType
from cli.shared.utils.config import load_and_instantiate_config


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
