import click


@click.group()
def models():
    """
    Command to list available models.
    """


@models.command()
@click.option('-m', '--model-type',
              default='all',
              help="The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.")
def list(model_type: str):
    """Command to list available models.

    Args:

        model_type (str, optional): The type of model to list. 'custom' for custom models, 'standard' for standard models, and 'all' for all models. Default is 'all'.

    Raises:
        ValueError: If the model_type is not supported.
    """
    from cli.logic.utils.model import cli_models

    cli_models(model_type)


@models.command()
@click.option('-m', '--model-name', default=None, help='Name of the model to inspect (e.g. "resnet18").')
@click.option('-n', '--normalization', is_flag=True, type=click.BOOL, default=False, help='Whether to include normalization layer in the model summary.')
def layers(model_name: str, normalization: bool):
    """Command to list the layers of a model.

    Args:

        model_name (str): The name of the model (e.g. "resnet18").
        normalization (bool): Whether to include normalization layer in the model summary.

    Raises:
        ValueError: If the model name is not provided.
    """
    from cli.logic.utils.model import cli_model_layers

    cli_model_layers(model_name, normalization)
