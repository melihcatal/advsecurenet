import click


@click.group()
def evaluate():
    """
    Command to evaluate models.
    """


@evaluate.command()
@click.option('-c', '--config', type=click.Path(exists=True), default=None, help='Path to the evaluation configuration yml file.')
def evaluation(config: str, **kwargs):
    """
    Command to evaluate the model on adversarial examples.
    """
    from cli.logic.evaluation import cli_evaluation

    cli_evaluation(config, **kwargs)
