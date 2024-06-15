from enum import Enum


class DatasetType(Enum):
    """
    An enum class for the dataset types supported by the library. It is possible to use a custom dataset as well.
    """
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    SVHN = "SVHN"
    FASHION_MNIST = "FASHION_MNIST"
    IMAGENET = "IMAGENET"
    MNIST = "MNIST"
    CUSTOM = "CUSTOM"


class DataType(Enum):
    """
    An enum class for the data types.
    """
    TRAIN = "train"
    TEST = "test"
