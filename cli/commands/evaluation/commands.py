import click

from cli.commands.evaluation.adversarial.commands import adversarial
from cli.commands.evaluation.benign.commands import benign


@click.group()
def evaluate():
    """
    Command to evaluate models.
    """


evaluate.add_command(adversarial)
evaluate.add_command(benign)
