from unittest.mock import MagicMock, patch

import pytest

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.types.configs.dataloader_config import \
    DataLoaderConfig
from cli.shared.utils.dataloader import get_dataloader


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataloader.DataLoaderFactory.create_dataloader")
@patch("cli.shared.utils.dataloader.click.secho")
def test_get_dataloader_default(mock_secho, mock_create_dataloader):
    mock_config = MagicMock()
    mock_dataset = MagicMock(spec=BaseDataset)
    mock_loader_config = MagicMock()
    mock_config.default = mock_loader_config

    dataloader = get_dataloader(mock_config, mock_dataset)

    mock_create_dataloader.assert_called_once_with(
        DataLoaderConfig(
            dataset=mock_dataset,
            batch_size=mock_loader_config.batch_size,
            num_workers=mock_loader_config.num_workers,
            shuffle=mock_loader_config.shuffle,
            drop_last=mock_loader_config.drop_last,
            pin_memory=mock_loader_config.pin_memory,
            sampler=None
        )
    )
    assert dataloader == mock_create_dataloader()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataloader.DataLoaderFactory.create_dataloader")
@patch("cli.shared.utils.dataloader.click.secho")
def test_get_dataloader_train(mock_secho, mock_create_dataloader):
    mock_config = MagicMock()
    mock_dataset = MagicMock(spec=BaseDataset)
    mock_loader_config = MagicMock()
    mock_config.train = mock_loader_config

    dataloader = get_dataloader(
        mock_config, mock_dataset, dataset_type="train")

    mock_create_dataloader.assert_called_once_with(
        DataLoaderConfig(
            dataset=mock_dataset,
            batch_size=mock_loader_config.batch_size,
            num_workers=mock_loader_config.num_workers,
            shuffle=mock_loader_config.shuffle,
            drop_last=mock_loader_config.drop_last,
            pin_memory=mock_loader_config.pin_memory,
            sampler=None
        )
    )
    assert dataloader == mock_create_dataloader()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataloader.DataLoaderFactory.create_dataloader")
@patch("cli.shared.utils.dataloader.click.secho")
def test_get_dataloader_test(mock_secho, mock_create_dataloader):
    mock_config = MagicMock()
    mock_dataset = MagicMock(spec=BaseDataset)
    mock_loader_config = MagicMock()
    mock_config.test = mock_loader_config

    dataloader = get_dataloader(mock_config, mock_dataset, dataset_type="test")

    mock_create_dataloader.assert_called_once_with(
        DataLoaderConfig(
            dataset=mock_dataset,
            batch_size=mock_loader_config.batch_size,
            num_workers=mock_loader_config.num_workers,
            shuffle=mock_loader_config.shuffle,
            drop_last=mock_loader_config.drop_last,
            pin_memory=mock_loader_config.pin_memory,
            sampler=None
        )
    )
    assert dataloader == mock_create_dataloader()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataloader.DataLoaderFactory.create_dataloader")
@patch("cli.shared.utils.dataloader.click.secho")
@patch("torch.distributed.get_world_size", return_value=4)
@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_rank", return_value=0)
@patch("torch.utils.data.distributed.DistributedSampler", autospec=True)
@patch('cli.shared.utils.dataloader.logger')
def test_get_dataloader_ddp(mock_logger, mock_DistributedSampler, mock_get_rank, mock_is_initialized, mock_get_world_size, mock_secho, mock_create_dataloader):
    mock_config = MagicMock()
    mock_dataset = MagicMock(spec=BaseDataset)
    mock_loader_config = MagicMock()
    mock_config.train = mock_loader_config
    mock_loader_config.shuffle = True

    dataloader = get_dataloader(
        mock_config, mock_dataset, dataset_type="train", use_ddp=True)

    mock_logger.warning.assert_called_once_with(
        "Disabling shuffle for Distributed Data Parallel.")
    assert dataloader == mock_create_dataloader()


@pytest.mark.cli
@pytest.mark.essential
def test_get_dataloader_dataset_none():
    mock_config = MagicMock()
    with pytest.raises(ValueError, match="Dataset cannot be None"):
        get_dataloader(mock_config, None)
