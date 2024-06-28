import io
import logging
import pickle
import random
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pkg_resources
import requests
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset as TorchDataset

from advsecurenet.datasets.base_dataset import DatasetWrapper

logger = logging.getLogger(__name__)


class AdversarialTargetGenerator:
    """
    This module is responsible for generating target images. This is specially useful for targeted attacks and when the client doesn't provide a target image.
    The example can be found in `examples/attacks/targeted_attacks.ipynb`.
    """

    def generate_target_labels(self, data, targets: list = None, overwrite=False):
        """
        Generates target labels for the given data.

        Args:
            data (torch.utils.data.Dataset): The training data.
            targets (list, optional): The list of target labels. Defaults to None.
            overwrite (bool, optional): If True, overwrites the existing indices map. Defaults to False.

        Returns:
            list: The list of target labels.
        """
        # Step 1: Generate class to image indices map
        class_to_images = self._generate_class_to_image_indices_map(
            data, targets, overwrite)

        # Step 2: Shuffle and pair images across classes
        paired_images = self._shuffle_and_pair_images_across_classes(
            class_to_images)

        # Step 3: Extract target labels
        target_labels = [d['target_label'] for d in paired_images]

        return target_labels

    def generate_target_images_and_labels(self, data: TorchDataset, targets: Optional[List] = None, overwrite: Optional[bool] = False) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Generates target images and labels for the given data.

        Args:
            data (TorchDataset): The dataset to generate target images and labels from.
            targets (Optional[List]): The list of target labels. Defaults to None.
            overwrite (Optional[bool]): If True, overwrites the existing indices map. Defaults to False.

        Returns:
            Tuple[List[torch.Tensor], List[int]]: The list of target images and target labels.
        """
        # Step 1: Generate class to image indices map
        class_to_images = self._generate_class_to_image_indices_map(
            data, targets, overwrite)

        # Step 2: Shuffle and pair images across classes
        paired_images = self._shuffle_and_pair_images_across_classes(
            class_to_images)
        if not paired_images:
            return [], []

        # Step 3: Extract target labels
        target_labels = [d['target_label'] for d in paired_images]

        # Step 4: Extract target images
        target_img_indices = [x['target_image'] for x in paired_images]

        target_images_list = [data[idx][0].clone().detach()
                              for idx in target_img_indices]
        target_images = torch.stack(target_images_list)

        return target_images, target_labels

    def _save_images(self, paired_images, data, save_path="./generated_images"):
        """Saves the generated target images."""
        for i, pair in enumerate(paired_images):
            original_image, target_image = self._get_image_from_pair(
                pair, data)
            original_image.save(
                f"{save_path}/original_image_{i}.png")
            target_image.save(
                f"{save_path}/target_image_{i}.png")

    def _load_mappings(self, url):
        """Downloads and loads a pickle file from the provided URL."""
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            file_stream = io.BytesIO(response.content)
            return pickle.load(file_stream)
        else:
            raise ValueError(
                f"Failed to download the mappings from {url}. Please check the URL and try again.")

    def _get_existing_indices_map(self, data_type: str) -> Union[dict, None]:
        """Checks if the indices map already exists."""
        available_maps = {
            "imagenet": "https://advsecurenet.s3.eu-central-1.amazonaws.com/maps/imagenet_class_idx_to_img_indices.pkl",
        }
        try:
            if data_type in available_maps:
                mapping = self._load_mappings(available_maps[data_type])
                if mapping:
                    return mapping
            return None
        except ValueError as e:
            logger.warning(
                "Could not load the existing indices map. Detailed error: %s", str(e))
            return None

    def _extract_targets(self, data: Union[TorchDataset, DatasetWrapper, zip]) -> list:
        """Extracts the targets from the data."""

        # Helper function to attempt attribute access
        def get_attribute(dataset, attr):
            return getattr(dataset, attr, None)

        # Check for targets or labels in the dataset or its wrapped dataset
        for attr in ['targets', 'labels']:
            targets = get_attribute(data, attr)
            if targets is None and isinstance(data, DatasetWrapper):
                targets = get_attribute(data.dataset, attr)

            if targets is not None:
                return targets

        # Fallback: Extract targets by iterating over the dataset
        return [label for _, label in data]

    def _generate_class_to_image_indices_map(self, data, targets: list = None, overwrite=False) -> dict:
        """Generates a map of class to images."""
        # Check if the map already exists
        if not overwrite:
            existing_map = self._get_existing_indices_map(
                data.name)
            if existing_map:
                return existing_map

        # Extract targets from the data
        if targets is None:
            targets = self._extract_targets(data)

        class_idx_to_img_indices = defaultdict(list)
        for i, target in enumerate(targets):
            if isinstance(target, torch.Tensor):
                target = target.item()
            target = int(target)
            class_idx_to_img_indices[target].append(i)
        return class_idx_to_img_indices

    def _shuffle_and_pair_images_across_classes(self, class_to_images):
        """Pairs images from consecutive classes in a circular fashion and then shuffles the target pairs to break the circular pattern."""
        paired_images = []
        class_indices = sorted(class_to_images.keys())
        total_classes = len(class_indices)

        # Step 1: Create initial circular pairs
        for i, current_class in enumerate(class_indices):
            next_class = class_indices[(i + 1) % total_classes]
            current_class_images = class_to_images[current_class]
            next_class_images = class_to_images[next_class]

            for img_idx, original_image in enumerate(current_class_images):
                target_image_idx = img_idx % len(next_class_images)
                target_image = next_class_images[target_image_idx]

                paired_images.append({
                    'original_image': original_image,
                    'original_label': current_class,
                    'target_image': target_image,
                    'target_label': next_class
                })

        # Step 2: Shuffle the target pairs
        target_pairs = [(d['target_image'], d['target_label'])
                        for d in paired_images]
        random.shuffle(target_pairs)

        # Step 3: Reassign shuffled targets to break the circular pattern
        for i, d in enumerate(paired_images):
            if d['original_image'] == target_pairs[i][0] or d['original_label'] == target_pairs[i][1]:
                continue
            d['target_image'], d['target_label'] = target_pairs[i]

        # Step 4: Shuffle the entire list
        random.shuffle(paired_images)

        return paired_images

    def _validate_paired_images(self, paired_images):
        """Validates the paired images for consistency and correctness."""

        # Extracting images and labels
        original_images = [d['original_image'] for d in paired_images]
        original_labels = [d['original_label'] for d in paired_images]
        target_images = [d['target_image'] for d in paired_images]
        target_labels = [d['target_label'] for d in paired_images]

        # Zipping images and labels
        zipped_original = list(zip(original_images, original_labels))
        zipped_target = list(zip(target_images, target_labels))

        # Validating pairs
        for i, d in enumerate(paired_images):
            # Check if original and target images are not the same
            assert d['original_image'] != d[
                'target_image'], f"Error: Original and target images are the same for the following pair:{d}"

            # Check if zipped information matches with the paired data
            assert d['original_image'] == zipped_original[i][
                0], f"Error: Original image does not match with original image in zipped_original:{d}"
            assert d['original_label'] == zipped_original[i][
                1], f"Error: Original label does not match with original label in zipped_original:{d}"
            assert d['target_image'] == zipped_target[i][
                0], f"Error: Target image does not match with target image in zipped_target:{d}"
            assert d['target_label'] == zipped_target[i][
                1], f"Error: Target label does not match with target label in zipped_target:{d}"

            # Additional check if original and target images are the same (in this case, labels must also match)
            if d['original_image'] == d['target_image']:
                assert d['original_label'] == d[
                    'target_label'], f"Error: Original and target images are the same but labels are different: {d}"

    def _get_image_from_pair(self, pair, data):
        original_image = data[pair['original_image']][0]
        target_image = data[pair['target_image']][0]

        # Check if the images are already in PIL format, if not, convert
        if not isinstance(original_image, Image.Image):
            original_image = transforms.ToPILImage()(original_image)
        if not isinstance(target_image, Image.Image):
            target_image = transforms.ToPILImage()(target_image)

        return original_image, target_image

    def _show_image_pair(self, pair, data, labels=None):
        original_image, target_image = self._get_image_from_pair(
            pair, data)

        plt.subplot(1, 2, 1)
        # Assuming MNIST images, which are grayscale
        plt.imshow(original_image, cmap='gray')
        # Check if labels exist and use them in the title
        if labels is not None:
            plt.title(f"Original: {labels[pair['original_label']]}")
        else:
            plt.title(f"Original: {pair['original_label']}")

        plt.subplot(1, 2, 2)
        plt.imshow(target_image, cmap='gray')
        # Check if labels exist and use them in the title
        if labels is not None:
            plt.title(f"Target: {labels[pair['target_label']]}")
        else:
            plt.title(f"Target: {pair['target_label']}")

        plt.show()
