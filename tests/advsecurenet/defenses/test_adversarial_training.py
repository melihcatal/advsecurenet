from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
from advsecurenet.datasets.targeted_adv_dataset import AdversarialDataset
from advsecurenet.defenses.adversarial_training import AdversarialTraining
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import \
    AdversarialTrainingConfig


@pytest.fixture
def mock_model():
    class MockModel(BaseModel):
        def __init__(self):
            super(MockModel, self).__init__()
            self.model_name = "mock_model"

        def forward(self, x):
            return x

        def eval(self):
            pass

        def parameters(self):
            return iter([torch.tensor([1.0])])

        def to(self, device):
            return self

        def load_model(self) -> None:
            pass

        def models(self):
            return [self]

    return MockModel()


@pytest.fixture
def mock_attack():
    class MockAttack(AdversarialAttack):
        def __init__(self, name="mock_attack", targeted=False):
            self.name = name
            self.targeted = targeted

        def attack(self, model, data, target_labels, target_images=None):
            return data  # Simulate returning the adversarial examples

    return MockAttack()


@pytest.fixture
def mock_data_loader():
    return DataLoader(
        [(torch.randn(3, 32, 32), torch.tensor(1)) for _ in range(10)])


@pytest.fixture
def mock_config(mock_model, mock_attack, mock_data_loader):

    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model],
        attacks=[mock_attack],
        train_loader=mock_data_loader
    )

    return config


@pytest.fixture
def mock_adversarial_dataset_loader():
    base_dataset = torch.randn(3, 32, 32), torch.tensor(1)

    class MockAdversarialDataset(AdversarialDataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return torch.randn(3, 32, 32), torch.tensor(1)

    dataset = MockAdversarialDataset(
        base_dataset=base_dataset
    )
    return DataLoader(dataset)


@pytest.fixture
def adversarial_training(mock_config):
    return AdversarialTraining(mock_config)


@pytest.fixture
def images():
    return torch.randn(10, 3, 32, 32)


@pytest.fixture
def true_labels():
    return torch.randint(0, 10, (10,))


@pytest.fixture
def target_images():
    return torch.randn(10, 3, 32, 32)


@pytest.fixture
def target_labels():
    return torch.randint(0, 10, (10,))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_shuffle_data(adversarial_training):
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    target = torch.tensor([1, 2, 3])

    shuffled_data, shuffled_target = adversarial_training._shuffle_data(
        data, target)

    assert data.shape == shuffled_data.shape
    assert target.shape == shuffled_target.shape


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config(adversarial_training, mock_config):
    adversarial_training._check_config(mock_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_target_not_base_model(mock_model, mock_attack, mock_data_loader):
    config = AdversarialTrainingConfig(
        model=1,
        models=[mock_model],
        attacks=[mock_attack],
        train_loader=mock_data_loader
    )

    with pytest.raises(ValueError, match="Target model must be a subclass of BaseModel!"):
        AdversarialTraining(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_models_not_base_model(mock_model, mock_attack, mock_data_loader):
    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[1],
        attacks=[mock_attack],
        train_loader=mock_data_loader
    )

    with pytest.raises(ValueError, match="All models must be a subclass of BaseModel!"):
        AdversarialTraining(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_models_not_base_model_mix(mock_model, mock_attack, mock_data_loader):
    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model, 1],
        attacks=[mock_attack],
        train_loader=mock_data_loader
    )

    with pytest.raises(ValueError, match="All models must be a subclass of BaseModel!"):
        AdversarialTraining(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_attacks_not_adv(mock_model, mock_data_loader):
    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model],
        attacks=[1],
        train_loader=mock_data_loader
    )

    with pytest.raises(ValueError, match="All attacks must be a subclass of AdversarialAttack!"):
        AdversarialTraining(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_attacks_not_adv_mix(mock_model, mock_attack, mock_data_loader):
    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model],
        attacks=[1, mock_attack],
        train_loader=mock_data_loader
    )

    with pytest.raises(ValueError, match="All attacks must be a subclass of AdversarialAttack!"):
        AdversarialTraining(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_train_loader_not_instance_of_dataloader(mock_model, mock_attack):
    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model],
        attacks=[mock_attack],
        train_loader=1
    )

    with pytest.raises(ValueError, match="train_dataloader must be a DataLoader!"):
        AdversarialTraining(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_targeted_attack_with_non_adversarial_dataset(mock_model, mock_attack, mock_data_loader):
    targeted_attack = mock_attack
    targeted_attack.targeted = True

    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model],
        attacks=[targeted_attack],
        train_loader=mock_data_loader
    )

    with pytest.raises(ValueError, match="If any of the attacks are targeted, the train_loader dataset must be an instance of AdversarialDataset!"):
        AdversarialTraining(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_targeted_attack_with_adversarial_dataset(mock_model, mock_attack, mock_adversarial_dataset_loader):
    targeted_attack = mock_attack
    targeted_attack.targeted = True

    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model],
        attacks=[targeted_attack],
        train_loader=mock_adversarial_dataset_loader
    )

    try:
        AdversarialTraining(config)
    except ValueError:
        pytest.fail("Unexpected ValueError raised!")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_lots_attack_with_non_adversarial_dataset(mock_model, mock_attack, mock_data_loader):
    lots_attack = mock_attack
    lots_attack.name = "LOTS"

    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model],
        attacks=[lots_attack],
        train_loader=mock_data_loader
    )

    with pytest.raises(ValueError, match="If the LOTS attack is used, the train_loader dataset must be an instance of AdversarialDataset and must contain target images and target labels!"):
        AdversarialTraining(config)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_lots_attack_with_adversarial_dataset(mock_model, mock_attack, mock_adversarial_dataset_loader):
    lots_attack = mock_attack
    lots_attack.name = "LOTS"

    config = AdversarialTrainingConfig(
        model=mock_model,
        models=[mock_model],
        attacks=[lots_attack],
        train_loader=mock_adversarial_dataset_loader
    )

    try:
        AdversarialTraining(config)
    except ValueError:
        pytest.fail("Unexpected ValueError raised!")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_combine_clean_and_adversarial_data(adversarial_training):
    source = torch.randn(5, 3, 32, 32)
    adv_source = torch.randn(5, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])
    adv_targets = torch.tensor([5, 6, 7, 8, 9])

    combined_data, combined_targets = adversarial_training._combine_clean_and_adversarial_data(
        source, adv_source, targets, adv_targets
    )

    assert combined_data.shape == (10, 3, 32, 32)
    assert combined_targets.shape == (10,)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_pre_training(adversarial_training):
    adversarial_training._pre_training()
    for model in adversarial_training.config.models:
        assert model.training


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_pre_training_no_models(adversarial_training, mock_config):
    adversarial_training.config.models = []
    adversarial_training._pre_training()
    assert mock_config.model in adversarial_training.config.models


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.defenses.adversarial_training.AdversarialTraining._perform_attack', return_value=torch.randn(5, 3, 32, 32))
def test_generate_adversarial_batch(mock_perform_attack, adversarial_training):
    source = torch.randn(5, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    adv_source, adv_targets = adversarial_training._generate_adversarial_batch(
        source, targets)

    assert isinstance(adv_source, torch.Tensor)
    assert adv_source.shape == source.shape
    assert adv_targets.shape == targets.shape


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_move_to_device(adversarial_training):
    source = torch.randn(5, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    device_source, device_targets = adversarial_training._move_to_device(
        source, targets)

    assert device_source.device.type == adversarial_training._device
    assert device_targets.device.type == adversarial_training._device


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.defenses.adversarial_training.AdversarialTraining._perform_attack', return_value=torch.randn(3, 3, 32, 32))
@patch('advsecurenet.defenses.adversarial_training.AdversarialTraining._prepare_data', return_value=(torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])))
def test_run_epoch(mock_perform_attack, adversarial_training):
    adversarial_training._run_epoch(1)
    assert True  # If no exceptions are raised, the test passes.


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_train_loader(adversarial_training):
    loader = adversarial_training._get_train_loader(1)
    assert isinstance(loader, tqdm)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_prepare_data(adversarial_training):
    source = torch.randn(5, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4])

    prepared_source, prepared_targets = adversarial_training._prepare_data(
        source, targets)

    assert prepared_source.device.type == adversarial_training._device
    assert prepared_targets.device.type == adversarial_training._device


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_loss_divisor(adversarial_training):
    divisor = adversarial_training._get_loss_divisor()
    assert divisor == len(adversarial_training.config.train_loader)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_untargeted(mock_attack, mock_model, images, true_labels, adversarial_training):
    mock_attack.targeted = False

    with patch.object(mock_attack, 'attack', return_value=images) as mock_attack_func:
        result = adversarial_training._perform_attack(
            mock_attack, mock_model, images, true_labels)

        assert torch.equal(result, images)
        mock_attack_func.assert_called_once_with(
            mock_model, images, true_labels)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_targeted(mock_attack, mock_model, images, true_labels, target_labels, adversarial_training):
    mock_attack.targeted = True

    with patch.object(mock_attack, 'attack', return_value=images) as mock_attack_func:
        result = adversarial_training._perform_attack(
            mock_attack, mock_model, images, true_labels, target_labels=target_labels)

        assert torch.equal(result, images)
        mock_attack_func.assert_called_once_with(
            mock_model, images, target_labels)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_lots(mock_attack, mock_model, images, true_labels, target_images, target_labels, adversarial_training):
    mock_attack.targeted = True
    mock_attack.name = "LOTS"

    with patch.object(mock_attack, 'attack', return_value=images) as mock_attack_func:
        result = adversarial_training._perform_attack(
            mock_attack, mock_model, images, true_labels, target_images=target_images, target_labels=target_labels)

        assert torch.equal(result, images)
        mock_attack_func.assert_called_once_with(
            mock_model, images, target_labels, target_images)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(AdversarialTraining, '_get_train_loader')
@patch.object(AdversarialTraining, '_prepare_data')
@patch.object(AdversarialTraining, '_generate_adversarial_batch')
@patch.object(AdversarialTraining, '_combine_clean_and_adversarial_data')
@patch.object(AdversarialTraining, '_run_batch')
@patch.object(AdversarialTraining, '_get_loss_divisor', return_value=1)
@patch.object(AdversarialTraining, '_log_loss')
def test_run_epoch(mock_log_loss, mock_get_loss_divisor, mock_run_batch, mock_combine_clean_and_adversarial_data,
                   mock_generate_adversarial_batch, mock_prepare_data, mock_get_train_loader, adversarial_training):

    epoch = 1
    images = torch.randn(10, 3, 32, 32)
    true_labels = torch.randint(0, 10, (10,))
    target_images = torch.randn(10, 3, 32, 32)
    target_labels = torch.randint(0, 10, (10,))

    # Mock train loader to return batches of data
    mock_get_train_loader.return_value = [
        (images, true_labels, target_images, target_labels),
        (images, true_labels)
    ]

    # Mock prepare_data to return the same data
    mock_prepare_data.side_effect = lambda *args: args

    # Mock generate_adversarial_batch to return adversarial examples
    mock_generate_adversarial_batch.side_effect = lambda *args: (
        args[0], args[1])

    # Mock combine_clean_and_adversarial_data to return combined data
    mock_combine_clean_and_adversarial_data.side_effect = lambda *args: (
        args[0], args[2])

    # Mock run_batch to return a loss value
    mock_run_batch.return_value = 1.0

    adversarial_training._run_epoch(epoch)

    assert mock_run_batch.call_count == 2
    mock_log_loss.assert_called_once_with(epoch, 2.0)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(AdversarialTraining, '_get_train_loader')
@patch.object(AdversarialTraining, '_prepare_data')
@patch.object(AdversarialTraining, '_generate_adversarial_batch')
@patch.object(AdversarialTraining, '_combine_clean_and_adversarial_data')
@patch.object(AdversarialTraining, '_run_batch')
@patch.object(AdversarialTraining, '_get_loss_divisor', return_value=1)
@patch.object(AdversarialTraining, '_log_loss')
def test_run_epoch_no_target_data(mock_log_loss, mock_get_loss_divisor, mock_run_batch, mock_combine_clean_and_adversarial_data,
                                  mock_generate_adversarial_batch, mock_prepare_data, mock_get_train_loader, adversarial_training):

    epoch = 1
    images = torch.randn(10, 3, 32, 32)
    true_labels = torch.randint(0, 10, (10,))

    # Mock train loader to return batches of data without target images and labels
    mock_get_train_loader.return_value = [
        (images, true_labels),
        (images, true_labels)
    ]

    # Mock prepare_data to return the same data
    mock_prepare_data.side_effect = lambda *args: args

    # Mock generate_adversarial_batch to return adversarial examples
    mock_generate_adversarial_batch.side_effect = lambda *args: (
        args[0], args[1])

    # Mock combine_clean_and_adversarial_data to return combined data
    mock_combine_clean_and_adversarial_data.side_effect = lambda *args: (
        args[0], args[2])

    # Mock run_batch to return a loss value
    mock_run_batch.return_value = 1.0

    adversarial_training._run_epoch(epoch)

    assert mock_run_batch.call_count == 2
    mock_log_loss.assert_called_once_with(epoch, 2.0)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_prepare_data_none(adversarial_training):
    result = adversarial_training._prepare_data()
    assert result == []
