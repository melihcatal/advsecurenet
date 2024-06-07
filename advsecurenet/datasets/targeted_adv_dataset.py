from typing import Any, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from advsecurenet.datasets.base_dataset import BaseDataset


class AdversarialDataset(BaseDataset):
    """
    A dataset class that wraps a base dataset and allows for targeted adversarial attacks.

    Args:
        base_dataset (Dataset): The base dataset to wrap.
        target_labels (Optional[List[int]], optional): The target labels for the adversarial attack. Defaults to None.
        target_images (Optional[List[torch.Tensor]], optional): The target images for the adversarial attack. Defaults to None.
    """

    def __init__(self,
                 base_dataset: Dataset,
                 target_labels: Optional[Union[List[int],
                                               torch.Tensor]] = None,
                 target_images: Optional[Union[List[torch.Tensor], torch.Tensor]] = None):
        self.base_dataset = base_dataset

        self.target_labels = target_labels
        self.target_images = target_images

        # if target labels and base dataset have different lengths, raise an error
        if target_labels and len(target_labels) != len(base_dataset):
            raise ValueError(
                f"The target labels and the base dataset must have the same length. Target labels: {len(target_labels)}, Base dataset: {len(base_dataset)}"
            )
        if target_images and target_labels and len(target_images) != len(target_labels):
            raise ValueError(
                f"The target images and target labels must have the same length. Target images: {len(target_images)}, Target labels: {len(target_labels)}"
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int, int, Optional[torch.Tensor]]:
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[Any, int, int, Optional[torch.Tensor]]: A tuple containing the input data,
            the true label, the target label, and the target image (if available).
        """
        input_data, true_label = self.base_dataset[idx]
        target_label = self.target_labels[idx] if self.target_labels else true_label
        target_image = self.target_images[idx] if self.target_images else input_data
        return input_data, true_label, target_label, target_image

    def get_dataset_class(self):
        return self.base_dataset.get_dataset_class()
