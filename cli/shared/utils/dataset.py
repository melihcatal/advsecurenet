from typing import Optional, Tuple

from torch.utils.data import Dataset as TorchDataset

from advsecurenet.datasets import DatasetFactory
from advsecurenet.shared.types.dataset import DatasetType
from cli.shared.types.utils.dataset import DatasetCliConfigType


def get_datasets(config: DatasetCliConfigType) -> Tuple[Optional[TorchDataset], Optional[TorchDataset]]:
    """
    Load the datasets conditionally based on provided paths.

    Args:
        config (DatasetCliConfigType): Configuration for datasets.

    Returns:
        Tuple[Optional[TorchDataset], Optional[TorchDataset]]: Tuple containing the training dataset (if requested)
        and the testing dataset (if requested).
    """
    dataset_name = _validate_dataset_name(config.dataset_name)
    dataset_type = DatasetType(dataset_name)
    dataset_obj = DatasetFactory.create_dataset(
        dataset_type=dataset_type,
        preprocess_config=config.preprocessing,
    )

    try:
        train_data = dataset_obj.load_dataset(
            train=True,
            root=config.train_dataset_path,
            download=config.download)
    except FileNotFoundError:
        train_data = None

    try:
        test_data = dataset_obj.load_dataset(
            train=False,
            root=config.test_dataset_path,
            download=config.download)
    except FileNotFoundError:
        test_data = None

    return train_data, test_data


def _validate_dataset_name(dataset_name: str) -> str:
    """
    Validate the dataset name.

    Returns:
        str: The validated dataset name.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    dataset_name = dataset_name.upper()
    # TODO: Better way to validate dataset name
    # pylint: disable=protected-access
    if dataset_name not in DatasetType._value2member_map_:
        raise ValueError("Unsupported dataset name! Choose from: " +
                         ", ".join([e.value for e in DatasetType]))
    return dataset_name
