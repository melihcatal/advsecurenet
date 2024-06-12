import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
import torch

from advsecurenet.attacks.attacker import AttackerConfig
from advsecurenet.attacks.attacker.attacker import Attacker
from advsecurenet.attacks.attacker.ddp_attacker import DDPAttacker
from advsecurenet.attacks.gradient_based.fgsm import FGSM
from advsecurenet.datasets import DatasetFactory
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.attack_configs import FgsmAttackConfig
from advsecurenet.shared.types.configs.attack_configs.attacker_config import \
    AttackerConfig
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from advsecurenet.shared.types.configs.model_config import CreateModelConfig


@pytest.fixture
def attacker_config():
    device_cfg = DeviceConfig(
        processor="cuda",
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


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.cuda.device_count', return_value=2)
@patch('advsecurenet.attacks.attacker.ddp_attacker.set_visible_gpus')
@patch('advsecurenet.attacks.attacker.ddp_attacker.find_free_port', return_value=12345)
def test_ddp_attacker_initialization(mock_find_free_port, mock_set_visible_gpus, mock_device_count, attacker_config):
    gpu_ids = [0, 1]
    attacker = DDPAttacker(config=attacker_config, gpu_ids=gpu_ids)

    assert attacker._world_size == 2
    assert attacker._gpu_ids == gpu_ids
    mock_set_visible_gpus.assert_called_with(gpu_ids)
    assert os.environ['MASTER_ADDR'] == 'localhost'
    assert os.environ['MASTER_PORT'] == '12345'


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('torch.cuda.device_count', return_value=2)
@patch('advsecurenet.attacks.attacker.ddp_attacker.mp.spawn')
def test_execute_method(mock_spawn, attacker_config):
    gpu_ids = [0, 1]

    device_cfg = DeviceConfig(
        processor="cuda",
        use_ddp=True,
        gpu_ids=gpu_ids
    )

    cfg = AttackerConfig(
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

    attacker = DDPAttacker(config=cfg, gpu_ids=gpu_ids)
    result = attacker.execute()
    mock_spawn.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.attacks.attacker.ddp_attacker.dist.init_process_group')
@patch('advsecurenet.attacks.attacker.ddp_attacker.torch.cuda.set_device')
@patch('advsecurenet.attacks.attacker.ddp_attacker.DDP')
@patch('advsecurenet.attacks.attacker.ddp_attacker.DistributedEvalSampler')
def test_setup_method(mock_sampler, mock_ddp, mock_set_device, mock_init_pg, attacker_config):
    gpu_ids = [0, 1]
    attacker = DDPAttacker(config=attacker_config, gpu_ids=gpu_ids)

    rank = 0
    with patch.object(attacker, '_create_dataloader', return_value=MagicMock()) as mock_dataloader:
        attacker._setup(rank)

        mock_init_pg.assert_called_with(
            backend='nccl', rank=rank, world_size=2)
        mock_set_device.assert_called_with(torch.device('cuda:0'))
        mock_ddp.assert_called_once_with(attacker_config.model.to(
            torch.device('cuda:0')), device_ids=[torch.device('cuda:0')])
        mock_sampler.assert_called_once_with(
            attacker_config.dataloader.dataset)
        mock_dataloader.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('os.remove')
@patch('builtins.open', new_callable=mock_open)
@patch('advsecurenet.attacks.attacker.ddp_attacker.DDPAttacker._gather_results', return_value=[torch.tensor(1)])
def test_store_and_gather_results(mock_gather_results, mock_open, mock_remove, attacker_config):
    gpu_ids = [0, 1]
    attacker = DDPAttacker(config=attacker_config, gpu_ids=gpu_ids)

    adv_images = [torch.tensor(1)]
    attacker._rank = 0
    attacker._store_results(adv_images)

    mock_open.assert_called_once_with('./adv_images_0.pkl', 'wb')
    mock_open().write.assert_called_once()

    mock_open.reset_mock()
    gathered_images = attacker._gather_results()

    # mock_open.assert_called_with('./adv_images_0.pkl', 'rb')
    assert gathered_images == [torch.tensor(1)]
