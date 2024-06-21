from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision import datasets
from torchvision.transforms import v2 as transforms

from advsecurenet.datasets.base_dataset import (BaseDataset, DatasetWrapper,
                                                ImageFolderBaseDataset)
from advsecurenet.shared.types.configs.preprocess_config import (
    PreprocessConfig, PreprocessStep)
from advsecurenet.shared.types.dataset import DataType


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_image_folder_base_dataset_get_dataset_class():
    dataset = ImageFolderBaseDataset()
    assert dataset.get_dataset_class() == datasets.ImageFolder


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.datasets.base_dataset.datasets.ImageFolder.__init__', return_value=None)
def test_image_folder_base_dataset_create_dataset(mock_image_folder):
    dataset = ImageFolderBaseDataset()
    mock_transform = MagicMock()
    mock_root = "mock/root"
    dataset_instance = dataset._create_dataset(
        datasets.ImageFolder, mock_transform, mock_root)
    assert isinstance(dataset_instance, datasets.ImageFolder)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_dataset_wrapper():
    mock_dataset = [0, 1, 2, 3, 4]
    wrapper = DatasetWrapper(mock_dataset, "test_dataset")
    assert len(wrapper) == 5
    assert wrapper[0] == 0
    assert wrapper.name == "test_dataset"


class MockBaseDataset(BaseDataset):
    def get_dataset_class(self):
        return datasets.FakeData


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.datasets.base_dataset.pkg_resources.resource_filename', return_value='/mock/data')
@patch('advsecurenet.datasets.base_dataset.datasets.FakeData', autospec=True)
@patch('advsecurenet.datasets.base_dataset.BaseDataset._create_dataset')
def test_base_dataset_load_dataset(mock_create_dataset, mock_dataset_class, mock_resource_filename):
    mock_create_dataset.return_value = MagicMock()
    preprocess_config = PreprocessConfig()
    dataset = MockBaseDataset(preprocess_config)

    mock_transform = MagicMock()
    dataset.get_transforms = MagicMock(return_value=mock_transform)

    loaded_dataset = dataset.load_dataset(
        train=True, download=True)
    assert isinstance(loaded_dataset, DatasetWrapper)
    assert dataset.data_type == DataType.TRAIN


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_base_dataset_get_transforms():
    steps = [PreprocessStep(name="ToTensor")]
    preprocess_config = PreprocessConfig(steps=steps)
    dataset = MockBaseDataset(preprocess_config)

    transform = dataset.get_transforms()
    assert isinstance(transform, transforms.Compose)
    assert isinstance(transform.transforms[0], transforms.ToTensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_base_dataset_get_transforms_dict():
    steps = [{"name": "ToTensor"}]
    preprocess_config = PreprocessConfig(steps=steps)
    dataset = MockBaseDataset(preprocess_config)

    transform = dataset.get_transforms()
    assert isinstance(transform, transforms.Compose)
    assert isinstance(transform.transforms[0], transforms.ToTensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_base_dataset_get_transform():
    dataset = MockBaseDataset()
    transform = dataset._get_transform('ToTensor')
    assert transform == transforms.ToTensor

    with pytest.raises(ValueError):
        dataset._get_transform('NonExistentTransform')


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_to_preprocess_step_with_dicts():
    dataset = MockBaseDataset()
    input_steps = [{'name': 'step1', 'params': {'param1': 'value1'}}, {
        'name': 'step2', 'params': {'param2': 'value2'}}]
    result = dataset._to_preprocess_step(input_steps)
    assert all(isinstance(step, PreprocessStep) for step in result)
    assert result == [PreprocessStep(name='step1', params={
                                     'param1': 'value1'}), PreprocessStep(name='step2', params={'param2': 'value2'})]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_to_preprocess_step_with_preprocess_steps():
    dataset = MockBaseDataset()
    input_steps = [PreprocessStep(name='step1', params={
                                  'param1': 'value1'}), PreprocessStep(name='step2', params={'param2': 'value2'})]
    result = dataset._to_preprocess_step(input_steps)
    assert result == input_steps


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_to_preprocess_step_with_mixed_input():
    dataset = MockBaseDataset()
    input_steps = [{'name': 'step1', 'params': {'param1': 'value1'}},
                   PreprocessStep(name='step2', params={'param2': 'value2'})]
    result = dataset._to_preprocess_step(input_steps)
    assert all(isinstance(step, PreprocessStep) for step in result)
    assert result == [PreprocessStep(name='step1', params={
                                     'param1': 'value1'}), PreprocessStep(name='step2', params={'param2': 'value2'})]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_to_preprocess_step_with_empty_params():
    dataset = MockBaseDataset()
    input_steps = [{'name': 'step1'}, {'name': 'step2', 'params': None}]
    result = dataset._to_preprocess_step(input_steps)
    assert all(isinstance(step, PreprocessStep) for step in result)
    assert result == [PreprocessStep(
        name='step1', params=None), PreprocessStep(name='step2', params=None)]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_base_dataset_convert_param():
    dataset = MockBaseDataset()
    assert dataset._convert_param("torch.float32") == torch.float32
    with pytest.raises(ValueError):
        dataset._convert_param("not.a.torch.param")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.datasets.base_dataset.transforms.Compose', autospec=True)
def test_base_dataset_construct_transforms_dict(mock_transforms_compose):
    preprocess_steps = [{"name": "ToTensor"}]
    dataset = MockBaseDataset()
    dataset._get_transform = MagicMock(return_value=transforms.ToTensor)

    result = dataset._construct_transforms(preprocess_steps)
    assert len(result) == 1
    assert isinstance(result[0], transforms.ToTensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch('advsecurenet.datasets.base_dataset.transforms.Compose', autospec=True)
def test_base_dataset_construct_transforms(mock_transforms_compose):
    preprocess_steps = [PreprocessStep(name="ToTensor")]
    dataset = MockBaseDataset()
    dataset._get_transform = MagicMock(return_value=transforms.ToTensor)

    result = dataset._construct_transforms(preprocess_steps)
    assert len(result) == 1
    assert isinstance(result[0], transforms.ToTensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_base_dataset_available_transforms():
    dataset = MockBaseDataset()
    available_transforms = dataset._available_transforms()
    assert 'ToTensor' in available_transforms


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_base_dataset_len():
    dataset = MockBaseDataset()
    dataset._dataset = MagicMock()
    dataset._dataset.__len__.return_value = 100
    assert len(dataset) == 100

    dataset._dataset = None
    assert len(dataset) == 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_base_dataset_getitem():
    dataset = MockBaseDataset()
    dataset._dataset = [0, 1, 2, 3]
    assert dataset[1] == 1

    dataset._dataset = None
    with pytest.raises(NotImplementedError):
        dataset[0]
