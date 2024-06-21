from typing import Optional, Tuple, cast

from torch.utils.data import Dataset as TorchDataset

from advsecurenet.datasets import DatasetFactory
from advsecurenet.shared.types.dataset import DatasetType
from cli.shared.types.utils.dataset import (AttacksDatasetCliConfigType,
                                            DatasetCliConfigType)


def get_datasets(config: DatasetCliConfigType, **kwargs) -> Tuple[Optional[TorchDataset], Optional[TorchDataset]]:
    """
    Load the datasets conditionally based on provided paths.

    Args:
        config (DatasetCliConfigType): Configuration for datasets.
        **kwargs: Arbitrary keyword arguments for the dataset.

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

    def load_dataset_part(train: bool, path: Optional[str]) -> Optional[TorchDataset]:
        try:
            return dataset_obj.load_dataset(
                train=train,
                root=path,
                download=config.download,
                **kwargs
            )
        except FileNotFoundError:
            return None

    if isinstance(config, AttacksDatasetCliConfigType):
        config = cast(AttacksDatasetCliConfigType, config)
        train_data = load_dataset_part(
            train=True, path=config.train_dataset_path) if config.dataset_part in ["train", "all"] else None
        test_data = load_dataset_part(
            train=False, path=config.test_dataset_path) if config.dataset_part in ["test", "all"] else None
    else:
        train_data = load_dataset_part(
            train=True, path=config.train_dataset_path)
        test_data = load_dataset_part(
            train=False, path=config.test_dataset_path)

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
    try:
        DatasetType(dataset_name)
    except ValueError as e:
        raise ValueError("Unsupported dataset name! Choose from: " +
                         ", ".join([e.value for e in DatasetType])) from e

    return dataset_name
