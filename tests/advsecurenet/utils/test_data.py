import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from advsecurenet.utils.data import (get_subset_data, split_data,
                                     unnormalize_data)


class MockDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_subset_data():
    data = torch.arange(100).view(-1, 1).float()
    labels = torch.arange(100).long()
    dataset = MockDataset(data, labels)
    subset = get_subset_data(dataset, num_samples=10, random_seed=42)

    assert len(subset) == 10


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_unnormalize_data():
    data = torch.tensor([[[[0.0, 0.5], [1.0, 1.5]]]])
    mean = [0.5]
    std = [0.5]
    unnormalized_data = unnormalize_data(data, mean, std)
    expected_data = torch.tensor([[[[0.5, 0.75], [1.0, 1.25]]]])

    assert torch.allclose(unnormalized_data, expected_data, atol=1e-6)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_split_data():
    X = [[i] for i in range(100)]
    y = [i for i in range(100)]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.25, random_state=42)

    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20
    assert len(y_train) == 60
    assert len(y_val) == 20
    assert len(y_test) == 20

    # Check reproducibility with the same random_state
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split_data(
        X, y, test_size=0.2, val_size=0.25, random_state=42)

    assert X_train == X_train2
    assert X_val == X_val2
    assert X_test == X_test2
    assert y_train == y_train2
    assert y_val == y_val2
    assert y_test == y_test2


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_split_data_no_random_state():
    X = [[i] for i in range(100)]
    y = [i for i in range(100)]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.25)

    assert len(X_train) == 60
    assert len(X_val) == 20
    assert len(X_test) == 20
    assert len(y_train) == 60
    assert len(y_val) == 20
    assert len(y_test) == 20
