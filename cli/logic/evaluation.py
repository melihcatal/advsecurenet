from advsecurenet.shared.types.configs import ConfigType
from cli.types.evaluation import EvaluationCliConfigType
from cli.utils.config import load_and_instantiate_config


def cli_evaluation(config: str, **kwargs) -> None:
    """
    Logic function to execute evaluation.
    """

    config_data = load_and_instantiate_config(
        config=config,
        default_config_file="evaluation_config.yml",
        config_type=ConfigType.EVALUATION,
        config_class=EvaluationCliConfigType,
        **kwargs
    )
    print(config_data)
