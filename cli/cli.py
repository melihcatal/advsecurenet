"""
Command line interface for AdvSecureNet.
"""
import click

from .commands.attack import attack
from .commands.defense import defense
from .commands.evaluation import evaluation
from .commands.train import train
from .commands.utils import utils


@click.group()
@click.version_option(version='0.1.7')
def main():
    """
    Welcome to AdvSecureNet CLI!
    """


main.add_command(attack)
main.add_command(defense)
main.add_command(evaluation)
main.add_command(utils)
main.add_command(train)


if __name__ == "__main__":
    main()
