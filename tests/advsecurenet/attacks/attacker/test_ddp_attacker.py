import os
import pickle
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from advsecurenet.attacks.attacker import AttackerConfig
from advsecurenet.attacks.attacker.attacker import Attacker
from advsecurenet.attacks.attacker.ddp_attacker import DDPAttacker
from advsecurenet.attacks.gradient_based.fgsm import FGSM
from advsecurenet.datasets import DatasetFactory
from advsecurenet.distributed.ddp_base_task import DDPBaseTask
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.attack_configs import FgsmAttackConfig
from advsecurenet.shared.types.configs.attack_configs.attacker_config import \
    AttackerConfig
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from advsecurenet.shared.types.configs.model_config import CreateModelConfig


@pytest.fixture
def processor(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def attacker_config(processor):
    device_cfg = DeviceConfig(
        processor=processor,
        use_ddp=True,
        gpu_ids=[0, 1]
    )

    return AttackerConfig(
        model=ModelFactory.create_model(
            CreateModelConfig(
                model_name="CustomMnistModel",
                num_classes=10,
                num_input_channels=1,
                pretrained=False
            )
        ),
        attack=FGSM(config=FgsmAttackConfig(
            epsilon=0.3,
            device=device_cfg
        )),
        dataloader=DataLoaderConfig(
            # get the test dataset
            dataset=DatasetFactory.create_dataset(
                dataset_type="MNIST", return_loaded=True)[1]
        ),
        device=device_cfg
    )


@pytest.fixture
@patch('advsecurenet.attacks.attacker.ddp_attacker.DDPBaseTask._setup_model')
@patch('advsecurenet.attacks.attacker.ddp_attacker.DDPBaseTask._setup_device')
def ddp_attacker(mock_device, mock_model, attacker_config):
    rank = 0
    world_size = 2
    return DDPAttacker(config=attacker_config, rank=rank, world_size=world_size)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_ddp_attacker_initialization(ddp_attacker):
    assert ddp_attacker._rank == 0
    assert ddp_attacker._world_size == 2
    assert isinstance(ddp_attacker, DDPBaseTask)
    assert isinstance(ddp_attacker, Attacker)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.attacks.attacker.ddp_attacker.DDPBaseTask._setup_model')
@patch('advsecurenet.attacks.attacker.ddp_attacker.DDPBaseTask._setup_device')
def test_execute_attack(mock_device, mock_setup_model, attacker_config):
    attacker_config.return_adversarial_images = True

    ddp_attacker = DDPAttacker(config=attacker_config, rank=0, world_size=2)
    with mock.patch.object(ddp_attacker, '_execute_attack', return_value=['dummy_image']) as mock_execute_attack, \
            mock.patch.object(ddp_attacker, '_store_results') as mock_store_results:
        ddp_attacker.execute()
        mock_execute_attack.assert_called_once()
        mock_store_results.assert_called_once_with(['dummy_image'])


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_store_results(ddp_attacker):
    adv_images = ['dummy_image']
    output_path = f'./adv_images_{ddp_attacker._rank}.pkl'
    with mock.patch('builtins.open', mock.mock_open()) as mock_file, \
            mock.patch('pickle.dump') as mock_pickle_dump:
        ddp_attacker._store_results(adv_images)
        mock_file.assert_called_once_with(output_path, 'wb')
        mock_pickle_dump.assert_called_once_with(adv_images, mock_file())


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_gather_results():
    world_size = 2
    dummy_images = ['dummy_image']
    for rank in range(world_size):
        output_path = f'./adv_images_{rank}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(dummy_images, f)
    gathered_images = DDPAttacker.gather_results(world_size)
    assert gathered_images == dummy_images * world_size
    for rank in range(world_size):
        output_path = f'./adv_images_{rank}.pkl'
        assert not os.path.exists(output_path)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_iterator(ddp_attacker):
    iterator = ddp_attacker._get_iterator()
    if ddp_attacker._rank == 0:
        assert hasattr(iterator, '__iter__')
    else:
        assert not hasattr(iterator, '__iter__')
