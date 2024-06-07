from advsecurenet.evaluation.tester import Tester
from advsecurenet.shared.types.configs.test_config import TestConfig
from cli.shared.types.evaluation.testing import TestingCliConfigType
from cli.shared.utils.dataloader import get_dataloader
from cli.shared.utils.dataset import get_datasets
from cli.shared.utils.model import create_model


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
            processor=self.config.device.processor,
            topk=self.config.testing.topk,
        )

        tester = Tester(config)
        tester.test()
