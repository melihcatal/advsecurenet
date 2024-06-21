from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from advsecurenet.shared.types.configs.configs import ConfigType
from cli.logic.evaluation.evaluation import (cli_adversarial_evaluation,
                                             cli_list_adversarial_evaluations)
from cli.shared.types.evaluation import AdversarialEvaluationCliConfigType


@patch("cli.logic.evaluation.evaluation.load_and_instantiate_config")
@patch("cli.logic.evaluation.evaluation.CLIAdversarialEvaluator")
def test_cli_adversarial_evaluation(mock_CLIAdversarialEvaluator, mock_load_and_instantiate_config):
    mock_config_data = MagicMock()
    mock_load_and_instantiate_config.return_value = mock_config_data
    mock_evaluator_instance = mock_CLIAdversarialEvaluator.return_value

    cli_adversarial_evaluation("config_path", extra_arg="value")

    mock_load_and_instantiate_config.assert_called_once_with(
        config="config_path",
        default_config_file="adversarial_evaluation_config.yml",
        config_type=ConfigType.ADVERSARIAL_EVALUATION,
        config_class=AdversarialEvaluationCliConfigType,
        extra_arg="value"
    )
    mock_CLIAdversarialEvaluator.assert_called_once_with(
        mock_config_data, extra_arg="value")
    mock_evaluator_instance.run.assert_called_once()


@patch("cli.logic.evaluation.evaluation.click.secho")
def test_cli_list_adversarial_evaluations(mock_secho):
    with patch("cli.logic.evaluation.evaluation.adversarial_evaluators", {
        "eval1": MagicMock(),
        "eval2": MagicMock()
    }):
        cli_list_adversarial_evaluations()

    mock_secho.assert_any_call(
        "Available adversarial evaluation options:", fg="green", bold=True)
    mock_secho.assert_any_call(" - eval1", fg="cyan")
    mock_secho.assert_any_call(" - eval2", fg="cyan")
