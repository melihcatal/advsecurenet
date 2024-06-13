from unittest.mock import MagicMock, patch

import click
import pytest

from advsecurenet.shared.normalization_params import NormalizationParameters
from cli.logic.utils.normalization_params import (_validate_dataset_name,
                                                  cli_normalization_params)


@pytest.mark.cli
@pytest.mark.essential
@patch("click.echo")
def test_cli_normalization_params_no_dataset(mock_echo):
    datasets = ["CIFAR-10", "MNIST", "ImageNet"]
    with patch.object(NormalizationParameters, 'DATASETS', datasets):
        cli_normalization_params(None)

    # Check that the available datasets are printed
    mock_echo.assert_any_call("Available datasets:")
    for dataset in datasets:
        mock_echo.assert_any_call(f"- {dataset}")
    mock_echo.assert_any_call("")


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("advsecurenet.shared.normalization_params.NormalizationParameters.get_params")
def test_cli_normalization_params_valid_dataset(mock_get_params, mock_secho):
    dataset_name = "CIFAR-10"
    normalization_params = MagicMock()
    normalization_params.mean = [0.5, 0.5, 0.5]
    normalization_params.std = [0.2, 0.2, 0.2]

    mock_get_params.return_value = normalization_params

    cli_normalization_params(dataset_name)

    mock_get_params.assert_called_with(dataset_name)
    mock_secho.assert_any_call(
        f"Normalization parameters for {dataset_name}:", bold=True)
    mock_secho.assert_any_call(f"Mean: {normalization_params.mean}", bold=True)
    mock_secho.assert_any_call(
        f"Standard Deviation: {normalization_params.std}", bold=True)
    click.echo("")


@pytest.mark.cli
@pytest.mark.essential
@patch("advsecurenet.shared.normalization_params.NormalizationParameters.get_params", return_value=None)
def test_validate_dataset_name_invalid(mock_get_params):
    dataset_name = "UnsupportedDataset"
    with pytest.raises(click.ClickException):
        _validate_dataset_name(dataset_name)

    mock_get_params.assert_called_once_with(dataset_name)


@pytest.mark.cli
@pytest.mark.essential
@patch("advsecurenet.shared.normalization_params.NormalizationParameters.get_params")
def test_validate_dataset_name_valid(mock_get_params):
    dataset_name = "CIFAR-10"
    normalization_params = MagicMock()

    mock_get_params.return_value = normalization_params

    # Should not raise any exception
    _validate_dataset_name(dataset_name)

    mock_get_params.assert_called_once_with(dataset_name)
