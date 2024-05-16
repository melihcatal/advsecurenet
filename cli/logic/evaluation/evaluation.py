import click

from advsecurenet.shared.adversarial_evaluators import adversarial_evaluators
from advsecurenet.shared.types.configs import ConfigType
from cli.logic.evaluation.adversarial_evaluation.adversarial_evaluator import \
    CLIAdversarialEvaluator
from cli.shared.types.evaluation import AdversarialEvaluationCliConfigType
from cli.shared.utils.config import load_and_instantiate_config


def cli_adversarial_evaluation(config: str, **kwargs) -> None:
    """
    Logic function to execute evaluation.
    """

    config_data = load_and_instantiate_config(
        config=config,
        default_config_file="adversarial_evaluation_config.yml",
        config_type=ConfigType.ADVERSARIAL_EVALUATION,
        config_class=AdversarialEvaluationCliConfigType,
        **kwargs
    )
    evaluator = CLIAdversarialEvaluator(config_data, **kwargs)
    evaluator.run()


def cli_list_adversarial_evaluations():
    """
    Logic function to list available adversarial evaluation options.
    """
    click.secho("Available adversarial evaluation options:",
                fg="green", bold=True)
    keys = adversarial_evaluators.keys()
    for key in keys:
        click.secho(f" - {key}", fg="cyan")
