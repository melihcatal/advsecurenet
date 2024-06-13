from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Subset

from advsecurenet.shared.types.attacks import AttackType
from cli.logic.attack.attacker import CLIAttacker


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.get_datasets")
@patch("cli.logic.attack.attacker.create_model")
@patch("cli.logic.attack.attacker.read_data_from_file")
@patch("cli.logic.attack.attacker.save_images")
@patch("cli.logic.attack.attacker.DDPAttacker")
@patch("cli.logic.attack.attacker.Attacker")
@patch("cli.logic.attack.attacker.AdversarialTargetGenerator")
def test_cli_attacker_execute_single_gpu(mock_adv_target_gen, mock_attacker, mock_ddp_attacker, mock_save_images, mock_read_data, mock_create_model, mock_get_datasets):
    # Setup
    config = MagicMock()
    config.device.use_ddp = False
    config.device.gpu_ids = [0]
    config.attack_procedure.save_result_images = True
    config.attack_procedure.result_images_dir = "results"
    config.attack_procedure.result_images_prefix = "adv"
    mock_model = MagicMock()
    mock_create_model.return_value = mock_model

    attacker = CLIAttacker(config, AttackType.FGSM)

    # Mock methods
    mock_config = MagicMock()
    attacker._prepare_attack_config = MagicMock(return_value=mock_config)
    mock_attacker_instance = mock_attacker.return_value
    mock_attacker_instance.execute.return_value = ["image1", "image2"]

    # Execute
    with patch("click.secho") as mock_click_secho:
        attacker.execute()

    # Verify
    # mock_create_model.assert_called_once_with(config.model)
    mock_attacker.assert_called_once_with(config=mock_config)
    mock_attacker_instance.execute.assert_called_once()
    mock_save_images.assert_called_once_with(
        images=["image1", "image2"], path="results", prefix="adv")
    mock_click_secho.assert_called_once_with(
        "Attack completed successfully.", fg="green")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.get_datasets")
@patch("cli.logic.attack.attacker.create_model")
@patch("cli.logic.attack.attacker.read_data_from_file")
@patch("cli.logic.attack.attacker.save_images")
@patch("cli.logic.attack.attacker.DDPAttacker")
@patch("cli.logic.attack.attacker.Attacker")
@patch("cli.logic.attack.attacker.AdversarialTargetGenerator")
def test_cli_attacker_execute_ddp(mock_adv_target_gen, mock_attacker, mock_ddp_attacker, mock_save_images, mock_read_data, mock_create_model, mock_get_datasets):
    config = MagicMock()
    config.device.use_ddp = True
    config.device.gpu_ids = [0, 1]
    config.attack_procedure.save_result_images = True
    config.attack_procedure.result_images_dir = "results"
    config.attack_procedure.result_images_prefix = "adv"
    mock_model = MagicMock()
    mock_create_model.return_value = mock_model

    attacker = CLIAttacker(config, AttackType.FGSM)

    # Mock methods
    mock_config = MagicMock()
    attacker._prepare_attack_config = MagicMock(return_value=mock_config)
    mock_attacker_instance = mock_ddp_attacker.return_value
    mock_attacker_instance.execute.return_value = ["image1", "image2"]

    # Execute
    with patch("click.secho") as mock_click_secho:
        attacker.execute()

    # Verify
    # mock_create_model.assert_called_once_with(config.model)
    mock_ddp_attacker.assert_called_once_with(
        config=mock_config, gpu_ids=[0, 1])
    mock_attacker_instance.execute.assert_called_once()
    mock_save_images.assert_called_once_with(
        images=["image1", "image2"], path="results", prefix="adv")
    mock_click_secho.assert_called_once_with(
        "Attack completed successfully.", fg="green")


@pytest.mark.cli
@pytest.mark.essential
def test_cli_attacker_get_target_labels():
    config = MagicMock()
    config.attack_config.target_parameters.targeted = True
    config.attack_config.target_parameters.target_labels_path = "path/to/labels"
    config.attack_config.target_parameters.target_labels_separator = ","

    attacker = CLIAttacker(config, AttackType.FGSM)

    with patch("cli.logic.attack.attacker.read_data_from_file", return_value=[1, 2, 3]) as mock_read_data:
        target_labels = attacker._get_target_labels()
        mock_read_data.assert_called_once_with(
            file_path="path/to/labels", cast_type=int, return_type=list, separator=",")
        assert target_labels == [1, 2, 3]


@pytest.mark.cli
@pytest.mark.essential
def test_cli_attacker_get_target_labels_not_targeted():
    config = MagicMock()
    config.attack_config.target_parameters.targeted = False

    attacker = CLIAttacker(config, AttackType.FGSM)

    with patch("cli.logic.attack.attacker.read_data_from_file") as mock_read_data:
        target_labels = attacker._get_target_labels()
        mock_read_data.assert_not_called()
        assert target_labels is None


@pytest.mark.cli
@pytest.mark.essential
def test_cli_attacker_validate_dataset_availability():
    config = MagicMock()
    attacker = CLIAttacker(config, AttackType.FGSM)

    valid_dataset = MagicMock()
    invalid_dataset = None

    assert attacker._validate_dataset_availability(
        valid_dataset, "train") == valid_dataset

    with pytest.raises(ValueError, match="The dataset part 'test' is specified but no test data is available."):
        attacker._validate_dataset_availability(invalid_dataset, "test")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.random_split")
def test_cli_attacker_sample_data(mock_random_split):
    config = MagicMock()
    attacker = CLIAttacker(config, AttackType.FGSM)

    mock_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))

    mock_random_split.return_value = [
        Subset(mock_dataset, [0, 1, 2]), Subset(mock_dataset, [3, 4, 5])]
    sampled_data = attacker._sample_data(mock_dataset, sample_size=3)
    assert len(sampled_data) == 3
