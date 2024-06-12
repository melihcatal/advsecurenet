import pytest
from torch.optim import lr_scheduler

from advsecurenet.shared.scheduler import Scheduler


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_scheduler_enum_values():
    assert Scheduler.REDUCE_LR_ON_PLATEAU.value == lr_scheduler.ReduceLROnPlateau
    assert Scheduler.STEP_LR.value == lr_scheduler.StepLR
    assert Scheduler.MULTI_STEP_LR.value == lr_scheduler.MultiStepLR
    assert Scheduler.COSINE_ANNEALING_LR.value == lr_scheduler.CosineAnnealingLR
    assert Scheduler.CYCLIC_LR.value == lr_scheduler.CyclicLR
    assert Scheduler.ONE_CYCLE_LR.value == lr_scheduler.OneCycleLR
    assert Scheduler.COSINE_ANNEALING_WARM_RESTARTS.value == lr_scheduler.CosineAnnealingWarmRestarts
    assert Scheduler.LAMBDA_LR.value == lr_scheduler.LambdaLR
    assert Scheduler.POLY_LR.value == lr_scheduler.PolynomialLR
    assert Scheduler.LINEAR_LR.value == lr_scheduler.LinearLR
    assert Scheduler.EXPONENTIAL_LR.value == lr_scheduler.ExponentialLR


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_scheduler_enum_names():
    assert Scheduler.REDUCE_LR_ON_PLATEAU.name == "REDUCE_LR_ON_PLATEAU"
    assert Scheduler.STEP_LR.name == "STEP_LR"
    assert Scheduler.MULTI_STEP_LR.name == "MULTI_STEP_LR"
    assert Scheduler.COSINE_ANNEALING_LR.name == "COSINE_ANNEALING_LR"
    assert Scheduler.CYCLIC_LR.name == "CYCLIC_LR"
    assert Scheduler.ONE_CYCLE_LR.name == "ONE_CYCLE_LR"
    assert Scheduler.COSINE_ANNEALING_WARM_RESTARTS.name == "COSINE_ANNEALING_WARM_RESTARTS"
    assert Scheduler.LAMBDA_LR.name == "LAMBDA_LR"
    assert Scheduler.POLY_LR.name == "POLY_LR"
    assert Scheduler.LINEAR_LR.name == "LINEAR_LR"
    assert Scheduler.EXPONENTIAL_LR.name == "EXPONENTIAL_LR"
