import click

from cli.commands.utils.configs.commands import configs
from cli.commands.utils.models.commands import models
from cli.commands.utils.normalization.commands import normalization
from cli.commands.utils.weights.commands import weights


@click.group()
def utils():
    """
    Command to utilities.
    """


utils.add_command(weights)
utils.add_command(configs)
utils.add_command(models)
utils.add_command(normalization)
