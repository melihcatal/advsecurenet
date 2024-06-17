from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.shared.types.configs.attack_configs.attack_config import \
    AttackConfig
from advsecurenet.shared.types.configs.configs import ConfigType
from cli.logic.defense.adversarial_training.adversarial_training_cli import \
    ATCLITrainer
from cli.logic.train.trainer import CLITrainer
from cli.shared.types.utils.model import ModelCliConfigType
from cli.shared.utils.attack_mappings import attack_cli_mapping, attack_mapping


@pytest.fixture
def mock_prepare_dataset():
    with patch.object(CLITrainer, '_prepare_dataset', return_value=MagicMock()) as mock:
        yield mock


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.create_model")
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.load_and_instantiate_config")
def test_atcli_trainer_init(mock_load_and_instantiate_config, mock_create_model, mock_prepare_dataset):
    mock_config = MagicMock()
    mock_config.training = MagicMock()
    mock_config.adversarial_training = MagicMock()

    trainer = ATCLITrainer(mock_config)
    assert trainer.at_config == mock_config.adversarial_training


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.load_and_instantiate_config")
def test_atcli_trainer_prepare_attacks(mock_load_and_instantiate_config, mock_prepare_dataset):
    mock_config = MagicMock()
    mock_config.device = MagicMock()
    mock_at_config = MagicMock()
    mock_at_config.attacks = [{"PGD": "path/to/config"}]

    mock_load_and_instantiate_config.return_value = MagicMock(
        spec=AttackConfig)

    attack_type = MagicMock()
    attack_class = MagicMock()
    attack_type.value = attack_class
    attack_cli_mapping["PGD"] = (attack_type, MagicMock())
    attack_mapping["PGD"] = MagicMock()

    trainer = ATCLITrainer(mock_config)
    trainer.at_config = mock_at_config
    attacks = trainer._prepare_attacks()

    assert len(attacks) == 1
    mock_load_and_instantiate_config.assert_called_once_with(
        config="path/to/config",
        default_config_file="pgd_attack_base_config.yml",
        config_type=ConfigType.ATTACK,
        config_class=attack_mapping["PGD"]
    )
    attack_class.assert_called_once_with(
        mock_load_and_instantiate_config.return_value)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.create_model")
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.load_and_instantiate_config")
def test_atcli_trainer_prepare_models(mock_load_and_instantiate_config, mock_create_model, mock_prepare_dataset):
    mock_config = MagicMock()
    mock_model_config = {"config": "path/to/model/config"}
    mock_at_config = MagicMock()
    mock_at_config.models = [mock_model_config]

    mock_load_and_instantiate_config.return_value = MagicMock(
        spec=ModelCliConfigType)

    trainer = ATCLITrainer(mock_config)
    trainer.at_config = mock_at_config
    models = trainer._prepare_models()

    assert len(models) == 1
    mock_load_and_instantiate_config.assert_called_once_with(
        config="path/to/model/config",
        default_config_file="model_config.yml",
        config_type=ConfigType.MODEL,
        config_class=ModelCliConfigType
    )
    mock_create_model.assert_called_once_with(
        mock_load_and_instantiate_config.return_value)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.DDPAdversarialTraining")
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.load_and_instantiate_config")
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.create_model")
def test_atcli_trainer_ddp_training_fn(mock_create_model, mock_load_and_instantiate_config, mock_ddp_adversarial_training, mock_prepare_dataset):
    mock_config = MagicMock()
    mock_training_env = MagicMock()
    trainer = ATCLITrainer(mock_config)

    trainer._prepare_training_environment = MagicMock(
        return_value=mock_training_env)

    trainer._ddp_training_fn(0, 2)

    mock_ddp_adversarial_training.assert_called_once_with(
        mock_training_env, 0, 2)
    mock_ddp_adversarial_training.return_value.train.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.AdversarialTraining")
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.load_and_instantiate_config")
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.create_model")
def test_atcli_trainer_execute_training(mock_create_model, mock_load_and_instantiate_config, mock_adversarial_training, mock_prepare_dataset):
    mock_config = MagicMock()
    mock_training_env = MagicMock()
    trainer = ATCLITrainer(mock_config)

    trainer._prepare_training_environment = MagicMock(
        return_value=mock_training_env)

    trainer._execute_training()

    mock_adversarial_training.assert_called_once_with(mock_training_env)
    mock_adversarial_training.return_value.train.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.load_and_instantiate_config")
@patch("cli.logic.defense.adversarial_training.adversarial_training_cli.create_model")
@patch("click.secho")
def test_atcli_trainer_train(mock_click_secho, mock_create_model, mock_load_and_instantiate_config, mock_prepare_dataset):
    mock_config = MagicMock()
    trainer = ATCLITrainer(mock_config)

    trainer._execute_training = MagicMock()
    trainer._execute_ddp_training = MagicMock()

    trainer.train()

    mock_click_secho.assert_called_once_with(
        "Starting Adversarial Training", fg="green")

    if mock_config.device.use_ddp:
        trainer._execute_ddp_training.assert_called_once()
    else:
        trainer._execute_training.assert_called_once()
