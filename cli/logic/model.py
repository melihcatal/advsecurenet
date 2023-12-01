"""
CLI command functions related to models.
"""

import click
from requests.exceptions import HTTPError

from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.models.standard_model import StandardModel
from advsecurenet.utils.model_utils import download_weights
from cli.utils.model import get_models as _get_models


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

    weights = StandardModel.available_weights(model_name)
    click.echo(f"Available weights for {model_name}:")
    for weight in weights:
        click.echo(f"\t{weight.name}")


def cli_model_layers(model_name: str):
    """
    List layers of a model.
    """
    if not model_name:
        raise ValueError("Model name must be provided!")

    model = ModelFactory.create_model(model_name, num_classes=3)
    layer_names = model.get_layer_names()
    click.echo(f"Layers for {model_name}:")
    click.echo(f"{'Layer Name':<30}{'Layer Type':<30}")
    for layer_name in layer_names:
        layer_type = type(model.get_layer(layer_name)).__name__
        click.echo(f"{layer_name:<30}{layer_type:<30}")

    # send a warning to remind the user to add model prefix while using LOTS Attack
    click.echo(click.style(
        'ATTENTION: You might need to add model prefix while using LOTS Attack. I.e. model.fc1',
        bold=True))


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
        print(
            f"Model weights for {model_name} trained on {dataset_name} already exist at {save_path_print}!")
    except HTTPError as e:
        print(
            f"Model weights for {model_name} trained on {dataset_name} not found on remote server!")
    except Exception as e:
        print(
            f"Error downloading model weights for {model_name} trained on {dataset_name}!")
