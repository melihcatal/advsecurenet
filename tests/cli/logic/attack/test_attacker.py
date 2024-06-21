from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Subset

from advsecurenet.attacks.attacker.ddp_attacker import DDPAttacker
from advsecurenet.shared.types.attacks import AttackType
from cli.logic.attack.attacker import CLIAttacker


@pytest.fixture
def attacker_config():
    config = MagicMock()
    config.attack_config.target_parameters.targeted = False
    return config


@pytest.fixture
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def ddp_attacker(mock_prepare_dataset, attacker_config):
    with mock.patch('torch.distributed.init_process_group'), \
            mock.patch('torch.distributed.destroy_process_group'), \
            mock.patch('torch.distributed.get_rank', return_value=0), \
            mock.patch('torch.distributed.get_world_size', return_value=2), \
            mock.patch('torch.distributed.all_reduce'), \
            mock.patch('torch.distributed.barrier'), \
            mock.patch('torch.nn.parallel.DistributedDataParallel', autospec=True):
        rank = 0
        world_size = 2
        return CLIAttacker(attacker_config, AttackType.FGSM)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.get_datasets")
@patch("cli.logic.attack.attacker.create_model")
@patch("cli.logic.attack.attacker.read_data_from_file")
@patch("cli.logic.attack.attacker.save_images")
@patch("cli.logic.attack.attacker.DDPAttacker")
@patch("cli.logic.attack.attacker.Attacker")
@patch("cli.logic.attack.attacker.AdversarialTargetGenerator")
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def test_cli_attacker_execute_single_gpu(mock_prepare_dataset, mock_adv_target_gen, mock_attacker, mock_ddp_attacker, mock_save_images, mock_read_data, mock_create_model, mock_get_datasets, attacker_config):
    # Setup
    attacker_config.device.use_ddp = False
    attacker_config.device.gpu_ids = [0]
    attacker_config.attack_procedure.save_result_images = True
    attacker_config.attack_procedure.result_images_dir = "results"
    attacker_config.attack_procedure.result_images_prefix = "adv"
    mock_model = MagicMock()
    mock_create_model.return_value = mock_model

    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

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
@patch('torch.cuda.device_count', return_value=2)
@patch('cli.logic.attack.attacker.set_visible_gpus')
@patch('cli.logic.attack.attacker.DDPCoordinator')
@patch('cli.logic.attack.attacker.DDPAttacker.gather_results', return_value=['dummy_image'])
@patch('cli.logic.attack.attacker.CLIAttacker._save_adversarial_images')
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def test_cli_attacker_execute_ddp_attack(
        mock_prepare_dataset,
        mock_save_adv_images,
        mock_gather_results,
        mock_ddp_coordinator,
        mock_set_visible_gpus,
        mock_device_count,
        ddp_attacker):

    ddp_attacker._execute_ddp_attack()

    mock_device_count.assert_called_once()
    mock_set_visible_gpus.assert_called_once_with([0, 1])
    mock_ddp_coordinator.assert_called_once_with(
        ddp_attacker._ddp_attack_fn, 2)
    mock_ddp_coordinator.return_value.run.assert_called_once()
    mock_gather_results.assert_called_once_with(2)
    mock_save_adv_images.assert_called_once_with(['dummy_image'])


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def test_cli_attacker_get_target_labels(mock_prepare_dataset, attacker_config):
    attacker_config.attack_config.target_parameters.targeted = True
    attacker_config.attack_config.target_parameters.target_labels_path = "path/to/labels"
    attacker_config.attack_config.target_parameters.target_labels_separator = ","

    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    with patch("cli.logic.attack.attacker.read_data_from_file", return_value=[1, 2, 3]) as mock_read_data:
        target_labels = attacker._get_target_labels()
        mock_read_data.assert_called_once_with(
            file_path="path/to/labels", cast_type=int, return_type=list, separator=",")
        assert target_labels == [1, 2, 3]


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def test_cli_attacker_get_target_labels_not_targeted(mock_prepare_dataset, attacker_config):
    attacker_config.attack_config.target_parameters.targeted = False
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    with patch("cli.logic.attack.attacker.read_data_from_file") as mock_read_data:
        target_labels = attacker._get_target_labels()
        mock_read_data.assert_not_called()
        assert target_labels is None


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def test_cli_attacker_validate_dataset_availability(mock_prepare_dataset, attacker_config):
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    valid_dataset = MagicMock()
    invalid_dataset = None

    assert attacker._validate_dataset_availability(
        valid_dataset, "train") == valid_dataset

    with pytest.raises(ValueError, match="The dataset part 'test' is specified but no test data is available."):
        attacker._validate_dataset_availability(invalid_dataset, "test")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch("cli.logic.attack.attacker.random_split")
def test_cli_attacker_sample_data(mock_random_split, mock_prepare_dataset, attacker_config):

    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    mock_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))

    mock_random_split.return_value = [
        Subset(mock_dataset, [0, 1, 2]),
        Subset(mock_dataset, [3, 4, 5])
    ]

    sampled_data = attacker._sample_data(mock_dataset, sample_size=3)
    assert len(sampled_data) == 3
