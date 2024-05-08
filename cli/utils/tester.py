from advsecurenet.shared.types.configs.test_config import TestConfig
from advsecurenet.utils.tester import Tester
from cli.types.testing import TestingCliConfigType
from cli.utils.dataloader import get_dataloader
from cli.utils.dataset import get_datasets
from cli.utils.model import create_model


class CLITester:
    """
    Tester class for the CLI. This module parses the CLI arguments and tests the model.

    Args:
        config (TestingCliConfigType): The configuration for testing.

    Attributes:
        config (TestingCliConfigType): The configuration for testing.
    """

    def __init__(self, config: TestingCliConfigType):
        self.config: TestingCliConfigType = config

    def test(self) -> None:
        """
        The main testing function. This function parses the CLI arguments and executes the testing.
        """
        model = create_model(self.config.model)
        _, test_dataset = get_datasets(self.config.dataset)
        test_loader = get_dataloader(
            self.config.dataloader,
            dataset=test_dataset,
            dataset_type='test',)
        config = TestConfig(
            model=model,
            test_loader=test_loader,
            criterion=self.config.testing.criterion,
            device=self.config.device.device)

        tester = Tester(config)
        tester.test()