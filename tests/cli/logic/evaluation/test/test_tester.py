from unittest.mock import MagicMock, patch

from advsecurenet.shared.types.configs.test_config import TestConfig
from cli.logic.evaluation.test.tester import CLITester


@patch("cli.logic.evaluation.test.tester.create_model")
@patch("cli.logic.evaluation.test.tester.get_datasets")
@patch("cli.logic.evaluation.test.tester.get_dataloader")
@patch("cli.logic.evaluation.test.tester.Tester")
def test_clitester_test(
    mock_Tester, mock_get_dataloader, mock_get_datasets, mock_create_model
):
    # Setup
    mock_config = MagicMock()
    mock_model = MagicMock()
    mock_test_dataset = MagicMock()
    mock_test_loader = MagicMock()
    mock_create_model.return_value = mock_model
    mock_get_datasets.return_value = (MagicMock(), mock_test_dataset)
    mock_get_dataloader.return_value = mock_test_loader

    tester = CLITester(mock_config)
    tester.test()

    # Verify
    mock_create_model.assert_called_once_with(mock_config.model)
    mock_get_datasets.assert_called_once_with(mock_config.dataset)
    mock_get_dataloader.assert_called_once_with(
        mock_config.dataloader,
        dataset=mock_test_dataset,
        dataset_type="test",
    )
    mock_Tester.assert_called_once_with(
        TestConfig(
            model=mock_model,
            test_loader=mock_test_loader,
            criterion=mock_config.testing.criterion,
            processor=mock_config.device.processor,
            topk=mock_config.testing.topk,
        )
    )
    mock_Tester.return_value.test.assert_called_once()


def test_clitester_init():
    # Setup
    mock_config = MagicMock()

    # Initialize CLITester
    tester = CLITester(mock_config)

    # Verify
    assert tester.config == mock_config
