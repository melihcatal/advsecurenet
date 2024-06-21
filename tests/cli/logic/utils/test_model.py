import enum
from unittest.mock import MagicMock, patch

import click
import pytest
from requests.exceptions import HTTPError

from cli.logic.utils.model import (_get_models, cli_available_weights,
                                   cli_download_weights, cli_model_layers,
                                   cli_models)


class ResNet18_Weights(enum.Enum):
    IMAGENET1K_V1 = "IMAGENET1K_V1"


@pytest.mark.cli
@pytest.mark.essential
@patch("click.echo")
@patch("advsecurenet.models.model_factory.ModelFactory.available_models", return_value=["model1", "model2"])
def test_cli_models_all_models(mock_available_models, mock_echo):
    cli_models("all")

    mock_available_models.assert_called_once()
    mock_echo.assert_any_call("Available models:\n")
    mock_echo.assert_any_call("1. model1")
    mock_echo.assert_any_call("2. model2")
    mock_echo.assert_any_call("")


@pytest.mark.cli
@pytest.mark.essential
@patch("click.echo")
@patch("advsecurenet.models.standard_model.StandardModel.available_weights", return_value=[ResNet18_Weights.IMAGENET1K_V1])
def test_cli_available_weights(mock_available_weights, mock_echo):
    cli_available_weights("resnet18")

    mock_available_weights.assert_called_once_with("resnet18")
    mock_echo.assert_any_call(
        "Available weights for resnet18:")
    mock_echo.assert_any_call("\tIMAGENET1K_V1")


@pytest.mark.cli
@pytest.mark.essential
def test_cli_available_weights_no_model():
    with pytest.raises(click.ClickException):
        cli_available_weights("")


@pytest.mark.cli
@pytest.mark.essential
def test_cli_available_weights_invalid_model():
    with patch("click.echo") as mock_echo:
        with pytest.raises(click.ClickException):
            cli_available_weights("invalid_model")
            mock_echo.assert_called_once_with(
                "Could not find available weights for the specified model!"
            )


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("click.echo")
@patch("advsecurenet.models.model_factory.ModelFactory.create_model")
def test_cli_model_layers_no_normalization(mock_create_model, mock_echo, mock_secho):
    mock_model = MagicMock()
    mock_model.get_layer_names.return_value = ["layer1", "layer2"]

    # Correctly mock get_layer to return a type with a name attribute
    mock_layer = MagicMock()
    mock_layer.__class__.__name__ = "Conv2D"
    mock_model.get_layer.return_value = mock_layer

    mock_create_model.return_value = mock_model

    cli_model_layers("model1", add_normalization=False)

    mock_create_model.assert_called_once_with(model_name="model1")
    mock_secho.assert_called_once_with(
        "Layers of model1:", bold=True, fg="green")
    mock_echo.assert_any_call(f"{'Layer Name':<30}{'Layer Type':<30}")
    mock_echo.assert_any_call(f"{'layer1':<30}{'Conv2D':<30}")
    mock_echo.assert_any_call(f"{'layer2':<30}{'Conv2D':<30}")


@pytest.mark.cli
@pytest.mark.essential
@patch("click.secho")
@patch("click.echo")
@patch("advsecurenet.models.model_factory.ModelFactory.create_model")
def test_cli_model_layer_with_normalization(mock_create_model, mock_echo, mock_secho):
    mock_model = MagicMock()
    mock_model.get_layer_names.return_value = [
        "NormalizationLayer", "layer1", "layer2"]

    # Correctly mock get_layer to return a type with a name attribute
    mock_layer = MagicMock()
    mock_layer.__class__.__name__ = "Conv2D"
    normalization_layer = MagicMock()
    normalization_layer.__class__.__name__ = "NormalizationLayer"

    def get_layer_side_effect(layer_name):
        if layer_name == "NormalizationLayer":
            return normalization_layer
        return mock_layer

    mock_model.get_layer.side_effect = get_layer_side_effect

    mock_create_model.return_value = mock_model

    cli_model_layers("model1", add_normalization=True)

    mock_create_model.assert_called_once_with(model_name="model1")
    mock_secho.assert_called_once_with(
        "Layers of model1:", bold=True, fg="green")
    mock_echo.assert_any_call(f"{'Layer Name':<30}{'Layer Type':<30}")
    mock_echo.assert_any_call(
        f"{'NormalizationLayer':<30}{'NormalizationLayer':<30}")
    mock_echo.assert_any_call(f"{'layer1':<30}{'Conv2D':<30}")
    mock_echo.assert_any_call(f"{'layer2':<30}{'Conv2D':<30}")


@pytest.mark.cli
@pytest.mark.essential
def test_cli_model_layers_no_model():
    with pytest.raises(ValueError):
        cli_model_layers("")


@pytest.mark.cli
@pytest.mark.essential
@patch("click.echo")
@patch("cli.logic.utils.model.download_weights")
def test_cli_download_weights(mock_download_weights, mock_echo):
    cli_download_weights("model1", "dataset1", "filename", "save_path")

    mock_download_weights.assert_called_once_with(
        "model1", "dataset1", "filename", "save_path")
    mock_echo.assert_called_once_with(
        "Downloaded weights to save_path. You can now use them for training or evaluation!"
    )


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.utils.model.download_weights", side_effect=FileExistsError)
def test_cli_download_weights_file_exists_error(mock_download_weights, capsys):
    with pytest.raises(click.ClickException) as excinfo:
        cli_download_weights("model1", "dataset1", "filename", "save_path")
    assert str(
        excinfo.value) == "File filename already exists in the specified directory!"


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.utils.model.download_weights", side_effect=HTTPError)
def test_cli_download_weights_http_error(mock_download_weights, capsys):
    with pytest.raises(click.ClickException) as excinfo:
        cli_download_weights("model1", "dataset1", "filename", "save_path")
    assert str(
        excinfo.value) == "Could not download weights for model1 trained on dataset1!"


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.utils.model.download_weights", side_effect=Exception("General error"))
def test_cli_download_weights_general_error(mock_download_weights, capsys):
    with pytest.raises(click.ClickException) as excinfo:
        cli_download_weights("model1", "dataset1", "filename", "save_path")
    assert str(excinfo.value) == "An error occurred while downloading the weights!"


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.utils.model.download_weights")
def test_cli_download_weights_success(mock_download_weights, capsys):
    cli_download_weights("model1", "dataset1", "filename", "save_path")
    captured = capsys.readouterr()
    assert "Downloaded weights to save_path. You can now use them for training or evaluation!" in captured.out


@pytest.mark.cli
@pytest.mark.essential
def test_get_models_all():
    with patch("advsecurenet.models.model_factory.ModelFactory.available_models", return_value=["model1", "model2"]) as mock_available_models:
        result = _get_models("all")
        assert result == ["model1", "model2"]
        mock_available_models.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
def test_get_models_custom():
    with patch("advsecurenet.models.model_factory.ModelFactory.available_custom_models", return_value=["custom_model1", "custom_model2"]) as mock_available_custom_models:
        result = _get_models("custom")
        assert result == ["custom_model1", "custom_model2"]
        mock_available_custom_models.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
def test_get_models_standard():
    with patch("advsecurenet.models.model_factory.ModelFactory.available_standard_models", return_value=["standard_model1", "standard_model2"]) as mock_available_standard_models:
        result = _get_models("standard")
        assert result == ["standard_model1", "standard_model2"]
        mock_available_standard_models.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
def test_get_models_invalid():
    with pytest.raises(ValueError):
        _get_models("invalid_type")
