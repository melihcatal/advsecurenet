from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.attacks import AdversarialAttack
from advsecurenet.defenses.adversarial_training import AdversarialTraining
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import \
    AdversarialTrainingConfig


class MockModel(BaseModel):
    def __init__(self):
        super(MockModel, self).__init__()
        self.model_name = "MockModel"

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


class MockAttack(AdversarialAttack):
    def __init__(self):
        self.name = "MockAttack"
        self.targeted = False

    def attack(self, model, data, target):
        return data  # Return a single tensor


@pytest.fixture
def mock_config():
    model = MockModel()
    attack = MockAttack()
    train_loader = DataLoader(
        [(torch.randn(3, 32, 32), torch.tensor(1)) for _ in range(10)])

    config = AdversarialTrainingConfig(
        model=model,
        models=[model],
        attacks=[attack],
        train_loader=train_loader
    )

    return config


@pytest.fixture
def adversarial_training(mock_config):
    return AdversarialTraining(mock_config)


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
