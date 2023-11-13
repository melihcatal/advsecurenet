import torch
from torch.utils.data import TensorDataset, random_split

def split_data(X, y, test_size=0.2, val_size=0.25, random_state=None):
    """
    Splits data into train, validation and test sets with the given ratios.

    Args:
        X (list): List of features.
        y (list): List of targets.
        test_size (float): Ratio of test samples.
        val_size (float): Ratio of validation samples.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train (list): List of training features.
        X_val (list): List of validation features.
        X_test (list): List of test features.
        y_train (list): List of training targets.
        y_val (list): List of validation targets.
        y_test (list): List of test targets.
    
    """
    # Seed for reproducibility
    if random_state is not None:
        torch.manual_seed(random_state)

    # Convert arrays to torch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)

    # Calculate split sizes
    total_samples = len(X)
    test_samples = int(test_size * total_samples)
    val_samples = int(val_size * (total_samples - test_samples))
    train_samples = total_samples - test_samples - val_samples

    # Perform random splits
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_samples, val_samples, test_samples])

    # Separate features and targets for each split
    X_train = [x[0] for x in train_dataset]
    y_train = [x[1] for x in train_dataset]

    X_val = [x[0] for x in val_dataset]
    y_val = [x[1] for x in val_dataset]

    X_test = [x[0] for x in test_dataset]
    y_test = [x[1] for x in test_dataset]

    return X_train, X_val, X_test, y_train, y_val, y_test
