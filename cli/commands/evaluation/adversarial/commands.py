import click


@click.group()
def adversarial():
    """
    Command to evaluate models on adversarial examples.
    """


@adversarial.command()
def list():
    """
    Command to list available adversarial evaluation options.
    """
    from cli.logic.evaluation.evaluation import \
        cli_list_adversarial_evaluations

    cli_list_adversarial_evaluations()


@adversarial.command()
@click.option('-c', '--config', type=click.Path(exists=True), default=None, help='Path to the evaluation configuration yml file.')
def eval(config: str, **kwargs):
    """
    Command to evaluate the model on adversarial examples.

    Args:
        config (str, optional): Path to the evaluation configuration yml file.

    """
    from cli.logic.evaluation.evaluation import cli_adversarial_evaluation

    cli_adversarial_evaluation(config, **kwargs)
