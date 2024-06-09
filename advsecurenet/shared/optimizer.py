from enum import Enum

from torch import optim


class Optimizer(Enum):
    """
    Enum class representing different optimization algorithms.
    """
    SGD = optim.SGD
    ADAM = optim.Adam
    ADAMW = optim.AdamW
    RMS_PROP = optim.RMSprop
    ADAGRAD = optim.Adagrad
    ADAMAX = optim.Adamax
    ASGD = optim.ASGD
    LBFGS = optim.LBFGS
    R_PROP = optim.Rprop
