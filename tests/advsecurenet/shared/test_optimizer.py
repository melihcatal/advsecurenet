import pytest
from torch import optim

from advsecurenet.shared.optimizer import Optimizer


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_optimizer_enum_values():
    assert Optimizer.SGD.value == optim.SGD
    assert Optimizer.ADAM.value == optim.Adam
    assert Optimizer.ADAMW.value == optim.AdamW
    assert Optimizer.RMS_PROP.value == optim.RMSprop
    assert Optimizer.ADAGRAD.value == optim.Adagrad
    assert Optimizer.ADAMAX.value == optim.Adamax
    assert Optimizer.ASGD.value == optim.ASGD
    assert Optimizer.LBFGS.value == optim.LBFGS
    assert Optimizer.R_PROP.value == optim.Rprop


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_optimizer_enum_names():
    assert Optimizer.SGD.name == "SGD"
    assert Optimizer.ADAM.name == "ADAM"
    assert Optimizer.ADAMW.name == "ADAMW"
    assert Optimizer.RMS_PROP.name == "RMS_PROP"
    assert Optimizer.ADAGRAD.name == "ADAGRAD"
    assert Optimizer.ADAMAX.name == "ADAMAX"
    assert Optimizer.ASGD.name == "ASGD"
    assert Optimizer.LBFGS.name == "LBFGS"
    assert Optimizer.R_PROP.name == "R_PROP"
