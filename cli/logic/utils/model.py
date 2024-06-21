"""
CLI command functions related to models.
"""
import click
from requests.exceptions import HTTPError

from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.models.standard_model import StandardModel
from advsecurenet.utils.model_utils import download_weights
from advsecurenet.utils.normalization_layer import NormalizationLayer


def cli_models(model_type: str):
    """
    List available models.
    """

    model_list = _get_models(model_type)

    click.echo("Available models:\n")
    # show numbers too
    for i, model in enumerate(model_list):
        click.echo(f"{i+1}. {model}")

    # add space
    click.echo("")


def cli_available_weights(model_name: str):
    """
    List available weights for a model.
    """
    if not model_name:
        raise click.ClickException(
            "Model name must be provided! You can use the 'models' command to list available models.")
    try:
        weights = StandardModel.available_weights(model_name.lower())
        click.echo(f"Available weights for {model_name}:")
        for weight in weights:
            click.echo(f"\t{weight.name}")
    except ValueError as e:
        raise click.ClickException(
            "Could not find available weights for the specified model!") from e


def cli_model_layers(model_name: str, add_normalization: bool = False):
    """
    List layers of a model.

    Args:
        model_name (str): The name of the model.
        add_normalization (bool): Whether to add normalization layers.

    Raises:
        ValueError: If the model name is not provided.
    """
    if not model_name:
        raise ValueError("Model name must be provided!")

    model = ModelFactory.create_model(model_name=model_name.lower())
    if add_normalization:
        # add a dummy normalization layer to correctly display the model summary
        model.add_layer(NormalizationLayer(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]), position=0, inplace=True)
    layer_names = model.get_layer_names()
    click.secho(f"Layers of {model_name}:", bold=True, fg="green")
    click.echo(f"{'Layer Name':<30}{'Layer Type':<30}")
    for layer_name in layer_names:
        layer_type = type(model.get_layer(layer_name)).__name__
        click.echo(f"{layer_name:<30}{layer_type:<30}")


def cli_download_weights(model_name: str, dataset_name: str, filename: str, save_path: str):
    """
    Download weights for a model and dataset.
    """

    if not model_name or not dataset_name:
        raise ValueError("Please provide both model name and dataset name!")
    try:
        save_path_print = save_path if save_path else "weights directory"
        download_weights(model_name, dataset_name, filename, save_path)
        click.echo(
            f"Downloaded weights to {save_path_print}. You can now use them for training or evaluation!")
    except FileExistsError as e:
        raise click.ClickException(
            f"File {filename} already exists in the specified directory!") from e
    except HTTPError as e:
        raise click.ClickException(
            f"Could not download weights for {model_name} trained on {dataset_name}!") from e
    except Exception as e:
        raise click.ClickException(
            "An error occurred while downloading the weights!") from e


def _get_models(model_type: str) -> list[str]:
    """
    Returns a list of available models of the specified type.
    """
    model_list_getters = {
        "all": ModelFactory.available_models,
        "custom": ModelFactory.available_custom_models,
        "standard": ModelFactory.available_standard_models
    }

    model_list = model_list_getters.get(model_type, lambda: [])()
    if not model_list:
        raise ValueError("Unsupported model type!")
    return model_list
