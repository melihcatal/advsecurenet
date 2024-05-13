import click


@click.group()
def defense():
    """
    Command to execute defenses.
    """


@defense.command()
@click.option('-c', '--config', type=click.Path(exists=True), default=None, help='Path to the adversarial configuration yml file.')
def adversarial_training(config: str, **kwargs):
    """
    Command to execute adversarial training. It can be used to train a single model or an ensemble of models and attacks.

    Args:
        config (str, optional): Path to the adversarial training configuration yml file.

    Examples:
        >>> advsecurenet defense adversarial-training --config= ./adversarial_training_config.yml

    Notes:
        Because of the large number of arguments, it is mandatory to use a configuration file for adversarial training.

    """
    from cli.logic.defense.defense import cli_adversarial_training

    cli_adversarial_training(config, **kwargs)
