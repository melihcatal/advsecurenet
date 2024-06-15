"""
This module represents the normalization parameters of the supported datasets. Users can either use the default normalization parameters or provide their own.
"""

from typing import Optional

import click

from advsecurenet.shared.normalization_params import NormalizationParameters


def cli_normalization_params(dataset_name: Optional[str] = None) -> None:
    """
    Print the normalization parameters for a specified dataset.

    Args:
        dataset_name (Optional[str]): The name of the dataset. If the dataset is not supported, an error is raised. If no dataset is provided, all available datasets are listed

    Note:
        The normalization parameters are the mean and standard deviation values for each channel of the dataset.
        dataset_name is case-sensitive i.e. "CIFAR-10" is different from "cifar-10".

    """
    if dataset_name is None:
        _list_datasets()
        return
    dataset_name = dataset_name.upper()
    _validate_dataset_name(dataset_name)
    normalization_params = NormalizationParameters.get_params(dataset_name)
    click.secho(f"Normalization parameters for {dataset_name}:", bold=True)
    click.secho(f"Mean: {normalization_params.mean}", bold=True)
    click.secho(f"Standard Deviation: {normalization_params.std}", bold=True)
    click.echo("")


def _list_datasets():
    """
    List the available datasets.

    Returns:
        List[str]: The list of available datasets.

    """
    click.echo("Available datasets:")
    for dataset in NormalizationParameters.DATASETS:
        click.echo(f"- {dataset}")
    click.echo("")


def _validate_dataset_name(dataset_name: str) -> None:
    """
    Validate the dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Raises:
        ValueError: If the dataset name is not supported.

    """
    params = NormalizationParameters.get_params(dataset_name)
    if params is None:
        raise click.ClickException(
            f"Dataset '{dataset_name}' is not supported. Supported datasets are: {NormalizationParameters.list_datasets()}")
