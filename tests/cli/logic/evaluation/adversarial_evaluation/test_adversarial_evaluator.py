from unittest.mock import MagicMock, patch

from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.configs import ConfigType
from cli.logic.evaluation.adversarial_evaluation.adversarial_evaluator import (
    CLIAdversarialEvaluator,
)
from cli.shared.types.utils.model import ModelCliConfigType


@patch("cli.logic.evaluation.adversarial_evaluation.adversarial_evaluator.cli_attack")
@patch("cli.logic.evaluation.adversarial_evaluation.adversarial_evaluator.create_model")
@patch(
    "cli.logic.evaluation.adversarial_evaluation.adversarial_evaluator.load_and_instantiate_config"
)
def test_cli_adversarial_evaluator_run(
    mock_load_and_instantiate_config, mock_create_model, mock_cli_attack
):
    mock_config = MagicMock()
    mock_config.evaluation_config.attack.name.upper.return_value = "PGD"
    mock_config.evaluation_config.attack.config = "attack_config_path"
    mock_config.evaluation_config.target_models = [{"config": "model_config_path"}]

    mock_model = MagicMock(spec=BaseModel)
    mock_load_and_instantiate_config.return_value = mock_model
    mock_create_model.return_value = mock_model

    evaluator = CLIAdversarialEvaluator(mock_config)
    evaluator.run()

    mock_cli_attack.assert_called_once_with(
        "PGD",
        "attack_config_path",
        target_models=[mock_model],
        evaluators=mock_config.evaluation_config.evaluators,
    )


@patch("cli.logic.evaluation.adversarial_evaluation.adversarial_evaluator.create_model")
@patch(
    "cli.logic.evaluation.adversarial_evaluation.adversarial_evaluator.load_and_instantiate_config"
)
def test_cli_adversarial_evaluator_prepare_target_models(
    mock_load_and_instantiate_config, mock_create_model
):
    mock_config = MagicMock()
    mock_config.evaluation_config.target_models = [{"config": "model_config_path"}]

    mock_model = MagicMock(spec=BaseModel)
    mock_load_and_instantiate_config.return_value = mock_model
    mock_create_model.return_value = mock_model

    evaluator = CLIAdversarialEvaluator(mock_config)
    models = evaluator._prepare_target_models()

    assert len(models) == 1
    mock_load_and_instantiate_config.assert_called_once_with(
        config="model_config_path",
        default_config_file="model_config.yml",
        config_type=ConfigType.MODEL,
        config_class=ModelCliConfigType,
    )
    mock_create_model.assert_called_once_with(mock_model)


def test_cli_adversarial_evaluator_prepare_target_models_no_models():
    mock_config = MagicMock()
    mock_config.evaluation_config.target_models = []

    evaluator = CLIAdversarialEvaluator(mock_config)
    models = evaluator._prepare_target_models()

    assert models == []
