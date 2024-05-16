import click


@click.group()
def weights():
    """
    Command to model weights.
    """


@weights.command()
@click.option('-m', '--model-name', default=None, help='Name of the model to evaluate (e.g. "resnet18").')
def list(model_name: str):
    """
    Command to list available weights for a model.

    Args:
        model_name (str): The name of the model (e.g. "resnet18").

    Raises:
        ClickException: If the model name is not provided.,

    Examples:
        >>> advsecurenet available-weights --model-name=resnet18
            IMAGENET1K_V1

    """
    from cli.logic.utils.model import cli_available_weights

    cli_available_weights(model_name)


@weights.command()
@click.option('--model-name', default=None, help='Name of the model for which weights are to be downloaded (e.g. "resnet18").')
@click.option('--dataset-name', default=None, help='Name of the dataset the model was trained on (e.g. "cifar10").')
@click.option('--filename', default=None, help='The filename of the weights on the remote server. If provided, this will be used directly.')
@click.option('--save-path', default=None, help='The directory to save the weights to. If not specified, defaults to the weights directory.')
def download(model_name, dataset_name, filename, save_path):
    """Command to download model weights from a remote source based on the model and dataset names.

    Args: 
        model_name (str, optional): The name of the model (e.g. "resnet18").
        dataset_name (str, optional): The name of the dataset the model was trained on (e.g. "cifar10").
        filename (str, optional): The filename of the weights on the remote server. If provided, this will be used directly.
        save_path (str, optional): The directory to save the weights to. Defaults to weights directory.
    """
    from cli.logic.utils.model import cli_download_weights

    cli_download_weights(model_name, dataset_name, filename, save_path)
