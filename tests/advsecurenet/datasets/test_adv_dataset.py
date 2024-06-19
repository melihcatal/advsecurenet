
import pytest
import torch

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.datasets.targeted_adv_dataset import AdversarialDataset


class MockBaseDataset(BaseDataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def get_dataset_class(self):
        return "MockBaseDataset"


@pytest.fixture
def base_dataset():
    data = [torch.randn(3, 32, 32) for _ in range(10)]
    targets = [i for i in range(10)]
    return MockBaseDataset(data, targets)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_dataset_initialization(base_dataset):
    # Test with valid target labels and images
    target_labels = [i for i in range(10)]
    target_images = [torch.randn(3, 32, 32) for _ in range(10)]
    adv_dataset = AdversarialDataset(
        base_dataset, target_labels=target_labels, target_images=target_images)
    assert len(adv_dataset) == len(base_dataset)

    # Test with invalid target labels length
    with pytest.raises(ValueError, match="The target labels and the base dataset must have the same length"):
        AdversarialDataset(base_dataset, target_labels=[1, 2, 3])

    # Test with invalid target images length
    with pytest.raises(ValueError, match="The target images and target labels must have the same length"):
        AdversarialDataset(base_dataset, target_labels=target_labels, target_images=[
                           torch.randn(3, 32, 32) for _ in range(5)])


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_dataset_len(base_dataset):
    adv_dataset = AdversarialDataset(base_dataset)
    assert len(adv_dataset) == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_dataset_getitem(base_dataset):
    target_labels = [i for i in range(10)]
    target_images = [torch.randn(3, 32, 32) for _ in range(10)]
    adv_dataset = AdversarialDataset(
        base_dataset, target_labels=target_labels, target_images=target_images)

    for i in range(10):
        images, true_labels, returned_target_images, returned_target_labels = adv_dataset[i]
        assert torch.equal(images, base_dataset[i][0])
        assert true_labels == base_dataset[i][1]
        assert torch.equal(returned_target_images, target_images[i])
        assert returned_target_labels == target_labels[i]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_dataset_getitem_no_targets(base_dataset):
    adv_dataset = AdversarialDataset(base_dataset)

    for i in range(10):
        images, true_labels, returned_target_images, returned_target_labels = adv_dataset[i]
        assert torch.equal(images, base_dataset[i][0])
        assert true_labels == base_dataset[i][1]
        assert torch.equal(returned_target_images, images)
        assert returned_target_labels == true_labels


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_adversarial_dataset_get_dataset_class(base_dataset):
    adv_dataset = AdversarialDataset(base_dataset)
    assert adv_dataset.get_dataset_class() == "MockBaseDataset"
