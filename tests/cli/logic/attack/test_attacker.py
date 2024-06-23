import logging
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Subset

from advsecurenet.attacks.attacker.ddp_attacker import DDPAttacker
from advsecurenet.attacks.gradient_based.fgsm import FGSM
from advsecurenet.datasets.targeted_adv_dataset import AdversarialDataset
from advsecurenet.shared.types.attacks import AttackType
from advsecurenet.shared.types.configs.attack_configs.attacker_config import \
    AttackerConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from cli.logic.attack.attacker import CLIAttacker
from cli.shared.types.attack import BaseAttackCLIConfigType

logger = logging.getLogger("cli.logic.attack.attacker")


@pytest.fixture
def device(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def attacker_config(device):
    device_config = DeviceConfig(
        processor=device,
        use_ddp=False
    )
    attacker_config = MagicMock()
    attacker_config.attack_config.target_parameters.targeted = False
    attacker_config.device = device_config
    return attacker_config


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


@pytest.fixture
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
def attacker(mock_prepare_dataset, attacker_config):
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
    # attacker
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
    # mock_create_model.assert_called_once_with(attacker_config.model)
    mock_attacker.assert_called_once_with(config=mock_config)
    mock_attacker_instance.execute.assert_called_once()
    mock_save_images.assert_called_once_with(
        images=["image1", "image2"], path="results", prefix="adv")
    mock_click_secho.assert_called_once_with(
        "Attack completed successfully.", fg="green")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.AdversarialTargetGenerator")
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch("cli.logic.attack.attacker.CLIAttacker._execute_ddp_attack")
def test_cli_attacker_execute_multi_gpus(mock_execute_ddp_attack, mock_prepare_dataset, mock_adv_target_gen, attacker_config):
    # attacker
    attacker_config.device.use_ddp = True
    attacker_config.device.gpu_ids = [0, 1]

    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    with mock.patch.object(attacker, '_execute_ddp_attack') as mock_execute_ddp_attack, \
            mock.patch.object(logger, 'info') as mock_logger_info:
        attacker.execute()

        mock_execute_ddp_attack.assert_called_once()
        mock_logger_info.assert_called()


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


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch('cli.logic.attack.attacker.AdversarialDataset')
def test_generate_target_labels_auto_generate(mock_adversarial_dataset, mock_prepare_dataset, attacker_config):
    attacker_config.attack_config.target_parameters.targeted = True
    attacker_config.attack_config.target_parameters.auto_generate_target = True
    attacker = CLIAttacker(attacker_config, AttackType.FGSM)

    all_data = MagicMock()
    target_parameters = MagicMock(targeted=True, auto_generate_target=True)
    target_labels = None

    with patch.object(attacker, '_generate_target', return_value=all_data) as mock_generate_target:
        result = attacker._generate_or_assign_target_labels(
            all_data, target_parameters, target_labels)
        mock_generate_target.assert_called_once_with(all_data)
        assert result == all_data


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch('cli.logic.attack.attacker.AdversarialDataset')
def test_generate_target_labels_with_labels(mock_adversarial_dataset, mock_prepare_dataset, attacker):
    all_data = MagicMock()
    target_parameters = MagicMock(targeted=True, auto_generate_target=False)
    target_labels = MagicMock()

    result = attacker._generate_or_assign_target_labels(
        all_data, target_parameters, target_labels)
    mock_adversarial_dataset.assert_called_once_with(
        base_dataset=all_data, target_labels=target_labels)
    assert result == mock_adversarial_dataset.return_value


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch('cli.logic.attack.attacker.AdversarialDataset')
def test_generate_target_labels_no_target_parameters(mock_adversarial_dataset, mock_prepare_dataset, attacker):
    all_data = MagicMock()
    target_parameters = None
    target_labels = None

    result = attacker._generate_or_assign_target_labels(
        all_data, target_parameters, target_labels)
    assert result == all_data
    mock_adversarial_dataset.assert_not_called()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_dataset")
@patch('cli.logic.attack.attacker.AdversarialDataset')
def test_generate_target_labels_not_targeted(mock_adversarial_dataset, mock_prepare_dataset, attacker):
    all_data = MagicMock()
    target_parameters = MagicMock(targeted=False)
    target_labels = None

    result = attacker._generate_or_assign_target_labels(
        all_data, target_parameters, target_labels)
    assert result == all_data
    mock_adversarial_dataset.assert_not_called()


@pytest.mark.cli
@pytest.mark.essential
@patch('cli.logic.attack.attacker.get_datasets', return_value=(MagicMock(), MagicMock()))
def test_prepare_dataset(mock_get_datasets, attacker):

    with patch.object(attacker, '_get_target_parameters', return_value=MagicMock()) as mock_get_target_parameters, \
            patch.object(attacker, '_get_target_labels_if_available', return_value=MagicMock()) as mock_get_target_labels_if_available, \
            patch.object(attacker, '_select_data_partition', return_value=MagicMock()) as mock_select_data_partition, \
            patch.object(attacker, '_sample_data_if_required', return_value=MagicMock()) as mock_sample_data_if_required, \
            patch.object(attacker, '_generate_or_assign_target_labels', return_value=MagicMock()) as mock_generate_or_assign_target_labels:

        dataset = attacker._prepare_dataset()

        mock_get_target_parameters.assert_called_once()
        mock_get_target_labels_if_available.assert_called_once_with(
            mock_get_target_parameters.return_value)
        mock_select_data_partition.assert_called_once()
        mock_sample_data_if_required.assert_called_once_with(
            mock_select_data_partition.return_value)
        mock_generate_or_assign_target_labels.assert_called_once_with(
            mock_sample_data_if_required.return_value, mock_get_target_parameters.return_value, mock_get_target_labels_if_available.return_value)
        assert dataset == mock_generate_or_assign_target_labels.return_value


@pytest.mark.cli
@pytest.mark.essential
def test_get_target_labels_if_available_with_labels(attacker):
    target_parameters = MagicMock()
    target_parameters.target_labels_config.target_labels_path = "some_path"
    target_parameters.target_labels_config.target_labels = None

    with patch.object(attacker, '_get_target_labels', return_value=MagicMock()) as mock_get_target_labels:
        target_labels = attacker._get_target_labels_if_available(
            target_parameters)
        mock_get_target_labels.assert_called_once()
        assert target_labels == mock_get_target_labels.return_value


@pytest.mark.cli
@pytest.mark.essential
def test_get_target_labels_if_available_without_labels(attacker):
    target_parameters = MagicMock()
    target_parameters.target_labels_config.target_labels_path = None
    target_parameters.target_labels_config.target_labels = None

    target_labels = attacker._get_target_labels_if_available(
        target_parameters)
    assert target_labels is None


@pytest.mark.cli
@pytest.mark.essential
def test_create_attack_targeted(attacker, attacker_config):
    attack_parameters = MagicMock()
    target_parameters = MagicMock(targeted=True)
    attacker._config.attack_config.attack_parameters = attack_parameters
    attacker._config.attack_config.target_parameters = target_parameters

    attack = attacker._create_attack()

    assert attack_parameters.targeted is True
    assert attack_parameters.device == attacker_config.device
    assert isinstance(attack, FGSM)


@pytest.mark.cli
@pytest.mark.essential
def test_create_attack_non_targeted(attacker, attacker_config):
    attack_parameters = MagicMock()
    target_parameters = MagicMock(targeted=None)
    attacker._config.attack_config.attack_parameters = attack_parameters
    attacker._config.attack_config.target_parameters = target_parameters

    attack = attacker._create_attack()

    assert attack_parameters.targeted is False
    assert attack_parameters.device == attacker_config.device
    assert isinstance(attack, FGSM)


@pytest.mark.cli
@pytest.mark.essential
def test_create_attack_missing_target_parameters(attacker, attacker_config):
    attack_parameters = MagicMock()
    attacker._config.attack_config.attack_parameters = attack_parameters
    attacker._config.attack_config.target_parameters = None

    attack = attacker._create_attack()

    assert attack_parameters.targeted is False
    assert attack_parameters.device == attacker_config.device
    assert isinstance(attack, FGSM)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker.__init__", return_value=None)
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_attack_config")
@patch("cli.logic.attack.attacker.DDPAttacker")
def test_ddp_attack_fn(mock_ddp_attacker, mock_prepare_attack_config, mock_init, attacker):
    mock_prepare_attack_config.return_value = MagicMock()
    mock_ddp_attacker.return_value = MagicMock()
    attacker._ddp_attack_fn(0, 2)

    mock_prepare_attack_config.assert_called_once()
    mock_ddp_attacker.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.CLIAttacker.__init__", return_value=None)
@patch("cli.logic.attack.attacker.CLIAttacker._prepare_attack_config")
@patch("cli.logic.attack.attacker.DDPAttacker")
def test_ddp_attack_fn(mock_ddp_attacker, mock_prepare_attack_config, mock_init, attacker):
    mock_prepare_attack_config.return_value = MagicMock()
    mock_ddp_attacker.return_value = MagicMock()
    attacker._ddp_attack_fn(0, 2)

    mock_prepare_attack_config.assert_called_once()
    mock_ddp_attacker.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.attacker.create_model")
@patch.object(CLIAttacker, '_create_attack')
@patch.object(CLIAttacker, '_create_dataloader_config')
def test_prepare_attack_config(mock_create_dataloader_config, mock_create_attack, mock_create_model, attacker, attacker_config):
    # Mock the return values of the methods
    mock_create_attack.return_value = "mock_attack"
    mock_create_model.return_value = "mock_model"
    mock_create_dataloader_config.return_value = "mock_dataloader_config"

    # Set up attacker_config attributes
    attacker_config.model = "mock_model_config"
    attacker_config.device = "mock_device"
    attacker_config.attack_procedure.save_result_images = True

    # Call the method to test
    result = attacker._prepare_attack_config()

    # Assert the correct creation of AttackerConfig
    assert isinstance(result, AttackerConfig)
    assert result.model == "mock_model"
    assert result.attack == "mock_attack"
    assert result.dataloader == "mock_dataloader_config"
    assert result.device == "mock_device"
    assert result.return_adversarial_images is True
    assert result.evaluators == ["attack_success_rate"]

    # Verify the mocks were called correctly
    mock_create_attack.assert_called_once()
    mock_create_model.assert_called_once_with("mock_model_config")
    mock_create_dataloader_config.assert_called_once()
