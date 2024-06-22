from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Subset

from advsecurenet.attacks.attacker.ddp_attacker import DDPAttacker
from advsecurenet.datasets.targeted_adv_dataset import AdversarialDataset
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


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def test_get_target_parameters(attacker_config):
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    target_parameters = attacker._get_target_parameters()
    assert target_parameters == attacker_config.attack_config.target_parameters


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def test_get_target_parameters_none(attacker_config):
    del attacker_config.attack_config.target_parameters
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    target_parameters = attacker._get_target_parameters()
    assert target_parameters is None


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch("cli.logic.attack.attacker.CLIAttacker._validate_dataset_availability")
@pytest.mark.parametrize("dataset_part", ["train", "test"])
def test_select_data_partition(mock_validate_dataset, mock_prepare_dataset, attacker_config, dataset_part):
    attacker_config.dataset.dataset_part = dataset_part
    train_data = MagicMock()
    test_data = MagicMock()
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)
    mock_validate_dataset.return_value = train_data if dataset_part == "train" else test_data

    returned_data = attacker._select_data_partition(train_data, test_data)

    mock_validate_dataset.assert_called_once_with(
        train_data if dataset_part == "train" else test_data, dataset_part)

    assert returned_data == (
        train_data if dataset_part == "train" else test_data)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch("cli.logic.attack.attacker.CLIAttacker._validate_dataset_availability")
def test_select_data_partition_only_test(mock_validate_dataset, mock_prepare_dataset, attacker_config):
    attacker_config.dataset.dataset_part = "all"
    test_data = MagicMock()
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    returned_data = attacker._select_data_partition(
        test_data=test_data, train_data=None)

    assert returned_data == test_data


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch("cli.logic.attack.attacker.CLIAttacker._validate_dataset_availability")
@pytest.mark.parametrize("sample_size", [None, 0, -1])
def test_sample_data_if_required_no_sampling(mock_validate_dataset, mock_prepare_dataset, attacker_config, sample_size):
    attacker_config.dataset.random_sample_size = sample_size
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    data = MagicMock()
    sampled_data = attacker._sample_data_if_required(data)

    assert sampled_data == data


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch("cli.logic.attack.attacker.CLIAttacker._validate_dataset_availability")
@patch("cli.logic.attack.attacker.CLIAttacker._sample_data")
@pytest.mark.parametrize("sample_size", [1, 10, 100])
def test_sample_data_if_required(mock_sample_data, mock_validate_dataset, mock_prepare_dataset, attacker_config, sample_size):
    attacker_config.dataset.random_sample_size = sample_size
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    data = MagicMock()
    sampled_data = attacker._sample_data_if_required(data)

    mock_sample_data.assert_called_once_with(data, sample_size)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch("cli.logic.attack.attacker.AdversarialTargetGenerator.generate_target_images_and_labels", return_value=([torch.randn(1, 3, 32, 32)], torch.tensor([1])))
def test_generate_target_lots(mock_generate_target_images, mock_prepare_dataset, attacker_config):
    # Create the CLIAttacker instance
    attacker = CLIAttacker(attacker_config, AttackType.LOTS)

    # Mock data to be passed to _generate_target
    data = MagicMock()
    data.__len__.return_value = 1

    # Call the method
    returned_data = attacker._generate_target(data)

    # Check if generate_target_images_and_labels was called once with the correct data
    mock_generate_target_images.assert_called_once_with(data=data)

    # Check the returned data
    assert isinstance(returned_data, AdversarialDataset)
    assert hasattr(returned_data, "target_images")
    assert hasattr(returned_data, "target_labels")
    assert len(returned_data.target_images) == 1
    assert returned_data.target_labels == torch.tensor([1])


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch("cli.logic.attack.attacker.AdversarialTargetGenerator.generate_target_labels", return_value=torch.tensor([1]))
def test_generate_target_fgsm(mock_generate_target_labels, mock_prepare_dataset, attacker_config):
    # Create the CLIAttacker instance
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    # Mock data to be passed to _generate_target
    data = MagicMock()
    data.__len__.return_value = 1

    # Call the method
    returned_data = attacker._generate_target(data)

    # Check if generate_target_images_and_labels was called once with the correct data
    mock_generate_target_labels.assert_called_once_with(
        data=data, overwrite=True)

    # Check the returned data
    assert isinstance(returned_data, AdversarialDataset)
    assert hasattr(returned_data, "target_labels")
    assert hasattr(returned_data, "target_images")
    assert returned_data.target_labels == torch.tensor([1])
    assert returned_data.target_images is None
