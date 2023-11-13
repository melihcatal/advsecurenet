from torch import optim
from enum import Enum

class Optimizer(Enum):
    SGD = optim.SGD
    ADAM = optim.Adam
    ADAMW = optim.AdamW
    RMS_PROP = optim.RMSprop
    ADAGRAD = optim.Adagrad
    ADAMAX = optim.Adamax
    ASGD = optim.ASGD
    LBFGS = optim.LBFGS
    R_PROP = optim.Rprop