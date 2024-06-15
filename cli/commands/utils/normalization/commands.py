import click


@click.group()
def normalization():
    """
    Command to list and get normalization values for the well-known datasets. The normalization values are the mean and standard deviation values for each channel of the dataset.
    """


@normalization.command()
def list():
    """Command to list available normalization values.

    Raises:
        ClickException: If no normalization values are found

    Examples:
        >>> advsecurenet normalization
    """
    from cli.logic.utils.normalization_params import _list_datasets

    _list_datasets()


@normalization.command()
@click.option('-d', '--dataset-name', default=None, help='Name of the dataset to inspect (e.g. "CIFAR10").')
def get(dataset_name: str):
    """Command to list the normalization values for a dataset.

    Args:

        dataset_name (str): The name of the dataset (e.g. "cifar10").

    Raises:
        ValueError: If the dataset name is not provided.
    """
    from cli.logic.utils.normalization_params import cli_normalization_params

    cli_normalization_params(dataset_name)
