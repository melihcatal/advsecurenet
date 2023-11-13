from enum import Enum


class DatasetType(Enum):
    """
    An enum class for the dataset types.
    """
    CIFAR10 = "CIFAR10"
    IMAGENET = "IMAGENET"
    MNIST = "MNIST"
    CUSTOM = "CUSTOM"


class DataType(Enum):
    """
    An enum class for the data types.
    """
    TRAIN = "train"
    TEST = "test"
