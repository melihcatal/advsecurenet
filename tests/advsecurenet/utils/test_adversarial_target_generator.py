import threading
import time
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest
import torch

from advsecurenet.datasets.base_dataset import DatasetWrapper
from advsecurenet.utils.adversarial_target_generator import \
    AdversarialTargetGenerator


@pytest.fixture
def mock_dataset():
    # Create a mock dataset with dummy images and labels
    data = [
        (torch.randn(3, 32, 32), torch.tensor(i % 10)) for i in range(100)
    ]
    return DatasetWrapper(data, name='mock_dataset')


@pytest.fixture
def target_generator():
    return AdversarialTargetGenerator()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_generate_target_labels(mock_dataset, target_generator):
    targets = target_generator.generate_target_labels(mock_dataset)
    assert isinstance(targets, list)
    assert len(targets) == len(mock_dataset)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_generate_target_images(mock_dataset, target_generator):
    targets = target_generator.generate_target_images(mock_dataset)
    assert isinstance(targets, list)
    assert len(targets) == len(mock_dataset)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_images_and_labels(mock_dataset, target_generator):
    paired_images = target_generator.generate_target_images(mock_dataset)
    original_images, original_labels, target_images, target_labels = target_generator.extract_images_and_labels(
        paired_images, mock_dataset)
    assert isinstance(original_images, torch.Tensor)
    assert isinstance(original_labels, torch.Tensor)
    assert isinstance(target_images, list)
    assert isinstance(target_labels, list)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.utils.adversarial_target_generator.requests.get')
@patch('advsecurenet.utils.adversarial_target_generator.pickle.load')
def test_load_mappings(mock_pickle_load, mock_requests_get, target_generator):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'test'
    mock_requests_get.return_value = mock_response
    mock_pickle_load.return_value = {'test': 'value'}
    result = target_generator._load_mappings('http://fakeurl.com')
    assert result == {'test': 'value'}


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_generate_class_to_image_indices_map(mock_dataset, target_generator):
    class_to_images = target_generator._generate_class_to_image_indices_map(
        mock_dataset)
    assert isinstance(class_to_images, dict)
    assert all(isinstance(key, int) for key in class_to_images.keys())
    assert all(isinstance(value, list) for value in class_to_images.values())


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_shuffle_and_pair_images_across_classes(mock_dataset, target_generator):
    class_to_images = target_generator._generate_class_to_image_indices_map(
        mock_dataset)
    paired_images = target_generator._shuffle_and_pair_images_across_classes(
        class_to_images)
    assert isinstance(paired_images, list)
    assert all(isinstance(pair, dict) for pair in paired_images)
    assert all(
        'original_image' in pair and 'target_image' in pair for pair in paired_images)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_validate_paired_images(mock_dataset, target_generator):
    class_to_images = target_generator._generate_class_to_image_indices_map(
        mock_dataset)
    paired_images = target_generator._shuffle_and_pair_images_across_classes(
        class_to_images)
    try:
        target_generator._validate_paired_images(paired_images)
    except AssertionError as e:
        pytest.fail(f"Validation failed: {e}")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.utils.adversarial_target_generator.Image.Image.save')
def test_save_images(mock_save, mock_dataset, target_generator):
    class_to_images = target_generator._generate_class_to_image_indices_map(
        mock_dataset)
    paired_images = target_generator._shuffle_and_pair_images_across_classes(
        class_to_images)
    target_generator._save_images(paired_images, mock_dataset, '/fakepath')
    assert mock_save.call_count == len(
        paired_images) * 2  # Each pair saves two images


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_show_image_pair(mock_dataset, target_generator):
    # Set the matplotlib backend to a non-interactive one for testing
    plt.switch_backend('Agg')
    class_to_images = target_generator._generate_class_to_image_indices_map(
        mock_dataset)
    paired_images = target_generator._shuffle_and_pair_images_across_classes(
        class_to_images)
    try:
        target_generator._show_image_pair(paired_images[0], mock_dataset)
    except Exception as e:
        pytest.fail(f"Show image pair failed: {e}")
