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
def mock_dataset_with_targets():
    # Create a mock dataset with 'targets' attribute
    data = [
        (torch.randn(3, 32, 32), torch.tensor(i % 10)) for i in range(100)
    ]
    dataset = DatasetWrapper(data, name='mock_dataset')
    dataset.targets = [i % 10 for i in range(100)]
    return dataset


@pytest.fixture
def mock_dataset_with_labels():
    # Create a mock dataset with 'labels' attribute
    data = [
        (torch.randn(3, 32, 32), torch.tensor(i % 10)) for i in range(100)
    ]
    dataset = DatasetWrapper(data, name='mock_dataset')
    dataset.labels = [i % 10 for i in range(100)]
    return dataset


@pytest.fixture
def mock_wrapped_dataset_with_targets():
    # Create a mock wrapped dataset
    data = [
        (torch.randn(3, 32, 32), torch.tensor(i % 10)) for i in range(100)
    ]
    inner_dataset = DatasetWrapper(data, name='inner_mock_dataset')
    inner_dataset.targets = [i % 10 for i in range(100)]
    outer_dataset = DatasetWrapper(inner_dataset, name='outer_mock_dataset')
    return outer_dataset


@pytest.fixture
def mock_wrapped_dataset_with_labels():
    # Create a mock wrapped dataset
    data = [
        (torch.randn(3, 32, 32), torch.tensor(i % 10)) for i in range(100)
    ]
    inner_dataset = DatasetWrapper(data, name='inner_mock_dataset')
    inner_dataset.labels = [i % 10 for i in range(100)]
    outer_dataset = DatasetWrapper(inner_dataset, name='outer_mock_dataset')
    return outer_dataset


@pytest.fixture
def mock_dataset_without_targets_or_labels():
    # Create a mock dataset without 'targets' or 'labels' attribute
    data = [
        (torch.randn(3, 32, 32), torch.tensor(i % 10)) for i in range(100)
    ]
    return DatasetWrapper(data, name='mock_dataset')


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
@patch('advsecurenet.utils.adversarial_target_generator.Image.Image.save')
def test_save_images_default_path(mock_save, mock_dataset, target_generator):
    class_to_images = target_generator._generate_class_to_image_indices_map(
        mock_dataset)
    paired_images = target_generator._shuffle_and_pair_images_across_classes(
        class_to_images)
    target_generator._save_images(paired_images, mock_dataset)
    assert mock_save.call_count == len(
        paired_images) * 2  # Each pair saves two images
    # Check that the default path is used
    assert 'generated_images' in mock_save.call_args[0][0]


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


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(AdversarialTargetGenerator, '_generate_class_to_image_indices_map')
@patch.object(AdversarialTargetGenerator, '_shuffle_and_pair_images_across_classes')
def test_generate_target_images_and_labels_success(mock_shuffle_and_pair, mock_generate_class_map, mock_dataset, target_generator):
    mock_generate_class_map.return_value = {
        i: list(range(i * 10, (i + 1) * 10)) for i in range(10)}
    mock_shuffle_and_pair.return_value = [
        {'target_label': i % 10, 'target_image': i} for i in range(20)
    ]

    target_images, target_labels = target_generator.generate_target_images_and_labels(
        mock_dataset)

    assert len(target_images) == 20
    assert len(target_labels) == 20
    assert all(isinstance(label, int) for label in target_labels)
    assert isinstance(target_images, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(AdversarialTargetGenerator, '_generate_class_to_image_indices_map')
@patch.object(AdversarialTargetGenerator, '_shuffle_and_pair_images_across_classes', return_value=[])
def test_generate_target_images_and_labels_empty_pairs(mock_shuffle_and_pair, mock_generate_class_map, mock_dataset, target_generator):
    mock_generate_class_map.return_value = {
        i: list(range(i * 10, (i + 1) * 10)) for i in range(10)}

    target_images, target_labels = target_generator.generate_target_images_and_labels(
        mock_dataset)

    assert len(target_images) == 0
    assert len(target_labels) == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_targets_with_targets(mock_dataset_with_targets, target_generator):
    targets = target_generator._extract_targets(mock_dataset_with_targets)
    assert targets == [i % 10 for i in range(100)]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_targets_with_labels(mock_dataset_with_labels, target_generator):
    targets = target_generator._extract_targets(mock_dataset_with_labels)
    assert targets == [i % 10 for i in range(100)]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_targets_wrapped_with_targets(mock_wrapped_dataset_with_targets, target_generator):
    targets = target_generator._extract_targets(
        mock_wrapped_dataset_with_targets)
    assert targets == [i % 10 for i in range(100)]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_targets_wrapped_with_labels(mock_wrapped_dataset_with_labels, target_generator):
    targets = target_generator._extract_targets(
        mock_wrapped_dataset_with_labels)
    assert targets == [i % 10 for i in range(100)]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_extract_targets_without_targets_or_labels(mock_dataset_without_targets_or_labels, target_generator):
    targets = target_generator._extract_targets(
        mock_dataset_without_targets_or_labels)
    expected_targets = [i % 10 for i in range(100)]
    assert targets == expected_targets


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(AdversarialTargetGenerator, '_load_mappings')
def test_get_existing_indices_map_success(mock_load_mappings, target_generator):
    data_type = "imagenet"
    mock_mapping = {'example_key': 'example_value'}
    mock_load_mappings.return_value = mock_mapping

    result = target_generator._get_existing_indices_map(data_type)

    assert result == mock_mapping
    mock_load_mappings.assert_called_once_with(
        "https://advsecurenet.s3.eu-central-1.amazonaws.com/maps/imagenet_class_idx_to_img_indices.pkl")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(AdversarialTargetGenerator, '_load_mappings')
def test_get_existing_indices_map_no_mapping(mock_load_mappings, target_generator):
    data_type = "imagenet"
    mock_load_mappings.return_value = None

    result = target_generator._get_existing_indices_map(data_type)

    assert result is None
    mock_load_mappings.assert_called_once_with(
        "https://advsecurenet.s3.eu-central-1.amazonaws.com/maps/imagenet_class_idx_to_img_indices.pkl")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(AdversarialTargetGenerator, '_load_mappings')
def test_get_existing_indices_map_invalid_data_type(mock_load_mappings, target_generator):
    data_type = "nonexistent_dataset"

    result = target_generator._get_existing_indices_map(data_type)

    assert result is None
    mock_load_mappings.assert_not_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(AdversarialTargetGenerator, '_load_mappings')
@patch('advsecurenet.utils.adversarial_target_generator.logger')
def test_get_existing_indices_map_load_failure(mock_logger, mock_load_mappings, target_generator):
    data_type = "imagenet"
    mock_load_mappings.side_effect = ValueError("Test exception")

    result = target_generator._get_existing_indices_map(data_type)

    assert result is None
    mock_load_mappings.assert_called_once_with(
        "https://advsecurenet.s3.eu-central-1.amazonaws.com/maps/imagenet_class_idx_to_img_indices.pkl")
    mock_logger.warning.assert_called_once_with(
        "Could not load the existing indices map. Detailed error: %s", "Test exception")
