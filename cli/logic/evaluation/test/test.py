import logging

from advsecurenet.shared.types.configs import ConfigType
from cli.logic.evaluation.test.tester import CLITester
from cli.shared.types.evaluation.testing import TestingCliConfigType
from cli.shared.utils.config import load_and_instantiate_config

logger = logging.getLogger(__name__)


def cli_test(config: str, **kwargs) -> None:
    """Main function for testing a model using the CLI.

    Args:
        config (str): The path to the configuration file.
        **kwargs: Additional keyword arguments.
    """
    config_data = load_and_instantiate_config(
        config=config,
        default_config_file="test_config.yml",
        config_type=ConfigType.TEST,
        config_class=TestingCliConfigType,
        **kwargs
    )

    logger.info("Loaded test configuration: %s", config_data)
    try:
        tester = CLITester(config_data)
        tester.test()
        logger.info("Model testing completed successfully")
    except Exception as e:
        logger.error("Failed to test model: %s", e)
        raise e
