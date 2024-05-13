import click

from cli.commands.utils.configs.commands import configs
from cli.commands.utils.models.commands import models
from cli.commands.utils.weights.commands import weights


@click.group()
def utils():
    """
    Command to utilities.
    """


utils.add_command(weights)
utils.add_command(configs)
utils.add_command(models)


@utils.command()
@click.option('-d', '--dataset-name', default=None, help='Name of the dataset to inspect (e.g. "CIFAR10").')
def normalization_params(dataset_name: str):
    """Command to list the normalization values for a dataset.

    Args:

        dataset_name (str): The name of the dataset (e.g. "cifar10").

    Raises:
        ValueError: If the dataset name is not provided.
    """
    from cli.logic.utils.normalization_params import cli_normalization_params

    cli_normalization_params(dataset_name)
