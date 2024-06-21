import pytest

from advsecurenet.shared.normalization_params import NormalizationParameters
from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.utils.dot_dict import DotDict


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_params_cifar10():
    params = NormalizationParameters.get_params(DatasetType.CIFAR10)
    assert isinstance(params, DotDict)
    assert params.mean == [0.4914, 0.4822, 0.4465]
    assert params.std == [0.2470, 0.2435, 0.2616]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_params_imagenet():
    params = NormalizationParameters.get_params(DatasetType.IMAGENET)
    assert isinstance(params, DotDict)
    assert params.mean == [0.485, 0.456, 0.406]
    assert params.std == [0.229, 0.224, 0.225]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_params_mnist():
    params = NormalizationParameters.get_params(DatasetType.MNIST)
    assert isinstance(params, DotDict)
    assert params.mean == [0.1307]
    assert params.std == [0.3081]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_params_invalid_string():
    with pytest.raises(KeyError):
        NormalizationParameters.get_params("INVALID")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_params_cifar10_string():
    params = NormalizationParameters.get_params("CIFAR10")
    assert isinstance(params, DotDict)
    assert params.mean == [0.4914, 0.4822, 0.4465]
    assert params.std == [0.2470, 0.2435, 0.2616]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_list_datasets():
    datasets = NormalizationParameters.list_datasets()
    assert isinstance(datasets, list)
    assert "CIFAR10" in datasets
    assert "IMAGENET" in datasets
    assert "MNIST" in datasets
