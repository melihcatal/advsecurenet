import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from advsecurenet.datasets.base_dataset import (BaseDataset,
                                                ImageFolderBaseDataset)
from advsecurenet.datasets.Cifar10.cifar10_dataset import (CIFAR10Dataset,
                                                           CIFAR100Dataset)
from advsecurenet.datasets.Custom.CustomDataset import CustomDataset
from advsecurenet.datasets.ImageNet.imagenet_dataset import ImageNetDataset
from advsecurenet.datasets.MNIST.mnist_dataset import (FashionMNISTDataset,
                                                       MNISTDataset)
from advsecurenet.datasets.svhn.svhn_dataset import SVHNDataset
from advsecurenet.shared.types.configs.preprocess_config import (
    PreprocessConfig, PreprocessStep)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_imagenet_dataset():
    # Instantiate the ImageNetDataset with an optional preprocess_config
    preprocess_steps = [PreprocessStep(
        name="Resize", params={"size": (256, 256)}), PreprocessStep(name="CenterCrop", params={"size": (224, 224)}
                                                                    )]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = ImageNetDataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.485, 0.456, 0.406], "Mean values do not match"
    assert dataset.std == [0.229, 0.224,
                           0.225], "Standard deviation values do not match"
    assert dataset.input_size == (256, 256), "Input size does not match"
    assert dataset.crop_size == (224, 224), "Crop size does not match"
    assert dataset.name == "imagenet", "Dataset name does not match"
    assert dataset.num_classes == 1000, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert dataset._preprocess_config == preprocess_config, "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(
        dataset, ImageFolderBaseDataset), "Instance is not of type ImageFolderBaseDataset"
    assert isinstance(
        dataset, BaseDataset), "Instance is not of type BaseDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_svhn_dataset():
    # Instantiate the SVHN with an optional preprocess_config
    preprocess_steps = [PreprocessStep(
        name="Resize", params={"size": (32, 32)}), PreprocessStep(name="CenterCrop", params={"size": (32, 32)}
                                                                  )]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = SVHNDataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.4377, 0.4438, 0.4728], "Mean values do not match"
    assert dataset.std == [0.1980, 0.2010,
                           0.1970], "Standard deviation values do not match"
    assert dataset.input_size == (32, 32), "Input size does not match"
    assert dataset.crop_size == (32, 32), "Crop size does not match"
    assert dataset.name == "svhn", "Dataset name does not match"
    assert dataset.num_classes == 10, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert dataset._preprocess_config == preprocess_config, "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(
        dataset, BaseDataset), "Instance is not of type BaseDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_mnist_dataset():
    # Instantiate the MNIST with an optional preprocess_config
    preprocess_steps = [PreprocessStep(
        name="Resize", params={"size": (28, 28)}), PreprocessStep(name="CenterCrop", params={"size": (28, 28)}
                                                                  )]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = MNISTDataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.1307], "Mean values do not match"
    assert dataset.std == [0.3081], "Standard deviation values do not match"
    assert dataset.input_size == (28, 28), "Input size does not match"
    assert dataset.crop_size == (28, 28), "Crop size does not match"
    assert dataset.name == "mnist", "Dataset name does not match"
    assert dataset.num_classes == 10, "Number of classes does not match"
    assert dataset.num_input_channels == 1, "Number of input channels does not match"
    assert dataset._preprocess_config == preprocess_config, "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(
        dataset, BaseDataset), "Instance is not of type BaseDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_fashion_mnist_dataset():
    # Instantiate the Fashion MNIST with an optional preprocess_config
    preprocess_steps = [PreprocessStep(
        name="Resize", params={"size": (28, 28)}), PreprocessStep(name="CenterCrop", params={"size": (28, 28)}
                                                                  )]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = FashionMNISTDataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.2860], "Mean values do not match"
    assert dataset.std == [0.3530], "Standard deviation values do not match"
    assert dataset.input_size == (28, 28), "Input size does not match"
    assert dataset.crop_size == (28, 28), "Crop size does not match"
    assert dataset.name == "fashion_mnist", "Dataset name does not match"
    assert dataset.num_classes == 10, "Number of classes does not match"
    assert dataset.num_input_channels == 1, "Number of input channels does not match"
    assert dataset._preprocess_config == preprocess_config, "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(
        dataset, BaseDataset), "Instance is not of type BaseDataset"
    assert isinstance(
        dataset, MNISTDataset), "Instance is not of type MNISTDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_cifar10_dataset():
    # Instantiate the CIFAR10 with an optional preprocess_config
    preprocess_steps = [PreprocessStep(
        name="Resize", params={"size": (32, 32)}), PreprocessStep(name="CenterCrop", params={"size": (32, 32)}
                                                                  )]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = CIFAR10Dataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.4914, 0.4822, 0.4465], "Mean values do not match"
    assert dataset.std == [0.2470, 0.2435,
                           0.2616], "Standard deviation values do not match"
    assert dataset.input_size == (32, 32), "Input size does not match"
    assert dataset.crop_size == (32, 32), "Crop size does not match"
    assert dataset.name == "cifar10", "Dataset name does not match"
    assert dataset.num_classes == 10, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert dataset._preprocess_config == preprocess_config, "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(
        dataset, BaseDataset), "Instance is not of type BaseDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_cifar100_dataset():
    # Instantiate the CIFAR100 with an optional preprocess_config
    preprocess_steps = [PreprocessStep(
        name="Resize", params={"size": (28, 28)}), PreprocessStep(name="CenterCrop", params={"size": (28, 28)}
                                                                  )]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = CIFAR100Dataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.5071, 0.4867, 0.4408], "Mean values do not match"
    assert dataset.std == [0.2675, 0.2565,
                           0.2761], "Standard deviation values do not match"
    assert dataset.input_size == (32, 32), "Input size does not match"
    assert dataset.crop_size == (32, 32), "Crop size does not match"
    assert dataset.name == "cifar100", "Dataset name does not match"
    assert dataset.num_classes == 100, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert dataset._preprocess_config == preprocess_config, "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(
        dataset, BaseDataset), "Instance is not of type BaseDataset"
    assert isinstance(
        dataset, CIFAR10Dataset), "Instance is not of type MNISTDataset"


@pytest.fixture
def temp_dataset_dir():
    # Create a temporary directory with the folder custom_dataset
    temp_dir = tempfile.mkdtemp()
    temp_dir = os.path.join(temp_dir, 'custom_dataset')

    # Create subdirectories and fake images
    os.makedirs(os.path.join(temp_dir, 'class1'))
    os.makedirs(os.path.join(temp_dir, 'class2'))

    img1 = Image.new('RGB', (60, 30), color='red')
    img2 = Image.new('RGB', (60, 30), color='blue')

    img1.save(os.path.join(temp_dir, 'class1', 'img1.jpg'))
    img2.save(os.path.join(temp_dir, 'class2', 'img2.jpg'))

    yield temp_dir

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_custom_dataset(temp_dataset_dir):
    preprocess_steps = [PreprocessStep(
        name="Resize", params={"size": (28, 28)}), PreprocessStep(name="CenterCrop", params={"size": (28, 28)}
                                                                  )]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)

    # Initialize the CustomDataset
    dataset = CustomDataset(preprocess_config=preprocess_config)
    # Set the root directory to the temp dataset directory
    dataset.root_dir = temp_dataset_dir

    dataset.load_dataset(root=temp_dataset_dir)
    # Assertions
    assert dataset.name == "custom", "Dataset name does not match"
    assert dataset._preprocess_config == preprocess_config, "Preprocess config does not match"

    # Verify the number of samples
    assert len(dataset) == 2, "Number of samples does not match"

    # Verify the labels
    assert dataset[0][1] == 0, "Label does not match"
    assert dataset[1][1] == 1, "Label does not match"
