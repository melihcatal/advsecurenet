from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import Dataset as TorchDataset

from advsecurenet.shared.types.dataset import DatasetType
from cli.shared.types.utils.dataset import (AttacksDatasetCliConfigType,
                                            DatasetCliConfigType)
from cli.shared.utils.dataset import _validate_dataset_name, get_datasets


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataset.DatasetFactory.create_dataset")
@patch("cli.shared.utils.dataset._validate_dataset_name", return_value="CIFAR10")
def test_get_datasets_standard(mock_validate_dataset_name, mock_create_dataset):
    mock_dataset = MagicMock()
    mock_create_dataset.return_value = mock_dataset
    mock_config = DatasetCliConfigType(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=MagicMock(),
        train_dataset_path="path/to/train",
        test_dataset_path="path/to/test",
        download=True
    )

    mock_dataset.load_dataset.side_effect = [
        MagicMock(spec=TorchDataset), MagicMock(spec=TorchDataset)]

    train_data, test_data = get_datasets(mock_config)

    mock_validate_dataset_name.assert_called_once_with("CIFAR10")
    mock_create_dataset.assert_called_once_with(
        dataset_type=DatasetType.CIFAR10, preprocess_config=mock_config.preprocessing)
    assert isinstance(train_data, TorchDataset)
    assert isinstance(test_data, TorchDataset)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataset.DatasetFactory.create_dataset")
@patch("cli.shared.utils.dataset._validate_dataset_name", return_value="CIFAR10")
def test_get_datasets_attacks(mock_validate_dataset_name, mock_create_dataset):
    mock_dataset = MagicMock()
    mock_create_dataset.return_value = mock_dataset
    mock_config = AttacksDatasetCliConfigType(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=MagicMock(),
        train_dataset_path="path/to/train",
        test_dataset_path="path/to/test",
        download=True,
        dataset_part="all"
    )

    mock_dataset.load_dataset.side_effect = [
        MagicMock(spec=TorchDataset), MagicMock(spec=TorchDataset)]

    train_data, test_data = get_datasets(mock_config)

    mock_validate_dataset_name.assert_called_once_with("CIFAR10")
    mock_create_dataset.assert_called_once_with(
        dataset_type=DatasetType.CIFAR10, preprocess_config=mock_config.preprocessing)
    mock_dataset.load_dataset.assert_any_call(
        train=True, root="path/to/train", download=True)
    mock_dataset.load_dataset.assert_any_call(
        train=False, root="path/to/test", download=True)
    assert isinstance(train_data, TorchDataset)
    assert isinstance(test_data, TorchDataset)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataset.DatasetFactory.create_dataset")
@patch("cli.shared.utils.dataset._validate_dataset_name", return_value="CIFAR10")
def test_get_datasets_file_not_found(mock_validate_dataset_name, mock_create_dataset):
    mock_dataset = MagicMock()
    mock_create_dataset.return_value = mock_dataset
    mock_config = DatasetCliConfigType(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=MagicMock(),
        train_dataset_path="path/to/train",
        test_dataset_path="path/to/test",
        download=True
    )

    mock_dataset.load_dataset.side_effect = FileNotFoundError

    train_data, test_data = get_datasets(mock_config)

    mock_validate_dataset_name.assert_called_once_with("CIFAR10")
    mock_create_dataset.assert_called_once_with(
        dataset_type=DatasetType.CIFAR10, preprocess_config=mock_config.preprocessing)
    assert train_data is None
    assert test_data is None


@pytest.mark.cli
@pytest.mark.essential
def test_validate_dataset_name_valid():
    assert _validate_dataset_name("CIFAR10") == "CIFAR10"


@pytest.mark.cli
@pytest.mark.essential
def test_validate_dataset_name_invalid():
    with pytest.raises(ValueError, match="Unsupported dataset name!"):
        _validate_dataset_name("INVALIDDATASET")
