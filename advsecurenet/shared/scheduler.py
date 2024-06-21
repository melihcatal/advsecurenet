from enum import Enum

from torch.optim import lr_scheduler


class Scheduler(Enum):
    """
    Supported schedulers for learning rate decay. Taken from https://pytorch.org/docs/stable/optim.html
    """
    REDUCE_LR_ON_PLATEAU = lr_scheduler.ReduceLROnPlateau
    STEP_LR = lr_scheduler.StepLR
    MULTI_STEP_LR = lr_scheduler.MultiStepLR
    COSINE_ANNEALING_LR = lr_scheduler.CosineAnnealingLR
    CYCLIC_LR = lr_scheduler.CyclicLR
    ONE_CYCLE_LR = lr_scheduler.OneCycleLR
    COSINE_ANNEALING_WARM_RESTARTS = lr_scheduler.CosineAnnealingWarmRestarts
    LAMBDA_LR = lr_scheduler.LambdaLR
    POLY_LR = lr_scheduler.PolynomialLR
    LINEAR_LR = lr_scheduler.LinearLR
    EXPONENTIAL_LR = lr_scheduler.ExponentialLR
