import torch

from advsecurenet.utils.data_splitter import split_data

def test_split_data():
    # Mock data
    X = torch.randn(100, 10)  # 100 samples with 10 features each
    y = torch.randint(0, 2, (100,))  # 100 binary targets

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, val_size=0.25, random_state=42)

    # Check split sizes
    assert len(X_train) == 60, "Incorrect number of training samples"
    assert len(X_val) == 20, "Incorrect number of validation samples"
    assert len(X_test) == 20, "Incorrect number of test samples"

    # Check if splits are tensors
    assert all(isinstance(x, torch.Tensor) for x in X_train), "Train features are not tensors"
    assert all(isinstance(x, torch.Tensor) for x in X_test), "Test features are not tensors"
    assert all(isinstance(y, torch.Tensor) for y in y_train), "Train labels are not tensors"
    assert all(isinstance(y, torch.Tensor) for y in y_test), "Test labels are not tensors"

    # Test reproducibility
    _, _, _, y_train2, _, _ = split_data(X, y, test_size=0.2, val_size=0.25, random_state=42)
    assert all(torch.equal(a, b) for a, b in zip(y_train, y_train2)), "Split is not reproducible with the same random state"
