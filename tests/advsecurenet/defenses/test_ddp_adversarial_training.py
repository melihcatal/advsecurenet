from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.defenses.ddp_adversarial_training import \
    DDPAdversarialTraining
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import \
    AdversarialTrainingConfig


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.defenses.ddp_adversarial_training.DDPTrainer.__init__')
@patch('advsecurenet.defenses.ddp_adversarial_training.AdversarialTraining.__init__')
def test_ddp_adversarial_training_init(mock_adversarial_training_init, mock_ddp_trainer_init):
    model = MagicMock()
    models = []
    attacks = [MagicMock()]
    config = AdversarialTrainingConfig(
        model=model, models=models, attacks=attacks)
    rank = 0
    world_size = 1

    DDPAdversarialTraining(config, rank, world_size)

    mock_ddp_trainer_init.assert_called_once()
    mock_adversarial_training_init.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(DDPAdversarialTraining, '__init__', return_value=None)
def test_get_train_loader(mock_ddp_adversarial_training_init):
    # Create mock config and dataset
    mock_sampler = MagicMock(spec=DistributedSampler)
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.sampler = mock_sampler

    mock_config = MagicMock(spec=AdversarialTrainingConfig)
    mock_config.train_loader = mock_train_loader

    # Initialize the DDPAdversarialTraining instance
    ddp_adversarial_training_instance = DDPAdversarialTraining.__new__(
        DDPAdversarialTraining)
    ddp_adversarial_training_instance.config = mock_config
    ddp_adversarial_training_instance._rank = 0

    # Test the _get_train_loader method
    epoch = 1
    with patch('advsecurenet.defenses.ddp_adversarial_training.tqdm', return_value=mock_train_loader) as mock_tqdm:
        train_loader = ddp_adversarial_training_instance._get_train_loader(
            epoch)

    # Assertions
    mock_sampler.set_epoch.assert_called_once_with(epoch)
    mock_tqdm.assert_called_once_with(mock_train_loader,
                                      desc="Adversarial Training",
                                      leave=False,
                                      position=1,
                                      unit="batch",
                                      colour="blue")
    assert train_loader == mock_train_loader


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(DDPAdversarialTraining, '__init__', return_value=None)
def test_get_train_loader_non_zero_rank(mock_ddp_adversarial_training_init):
    # Create mock config and dataset
    mock_sampler = MagicMock(spec=DistributedSampler)
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.sampler = mock_sampler

    mock_config = MagicMock(spec=AdversarialTrainingConfig)
    mock_config.train_loader = mock_train_loader

    # Initialize the DDPAdversarialTraining instance
    ddp_adversarial_training_instance = DDPAdversarialTraining.__new__(
        DDPAdversarialTraining)
    ddp_adversarial_training_instance.config = mock_config
    ddp_adversarial_training_instance._rank = 1

    # Test the _get_train_loader method
    epoch = 1
    train_loader = ddp_adversarial_training_instance._get_train_loader(epoch)

    # Assertions
    mock_sampler.set_epoch.assert_called_once_with(epoch)
    assert train_loader == mock_train_loader


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(DDPAdversarialTraining, '__init__', return_value=None)
def test_get_loss_divisor(mock_ddp_adversarial_training_init):
    # Create mock config and dataset
    mock_train_loader = MagicMock()
    # Mock the length of the train loader
    mock_train_loader.__len__.return_value = 10

    mock_config = MagicMock(spec=AdversarialTrainingConfig)
    mock_config.train_loader = mock_train_loader

    # Initialize the DDPAdversarialTraining instance
    ddp_adversarial_training_instance = DDPAdversarialTraining.__new__(
        DDPAdversarialTraining)
    ddp_adversarial_training_instance.config = mock_config
    ddp_adversarial_training_instance._world_size = 4

    # Test the _get_loss_divisor method
    loss_divisor = ddp_adversarial_training_instance._get_loss_divisor()

    # Assertions
    assert loss_divisor == 40, "Loss divisor calculation is incorrect"
