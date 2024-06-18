
import pytest
from torch import nn

from advsecurenet.utils.loss import get_loss_function


@pytest.mark.advsecurenet
@pytest.mark.essential
def get_loss_function_with_string():
    loss_fn = get_loss_function("cross_entropy")
    assert isinstance(loss_fn, nn.CrossEntropyLoss)


@pytest.mark.advsecurenet
@pytest.mark.essential
def get_loss_function_with_module():
    loss_fn = get_loss_function(nn.CrossEntropyLoss())
    assert isinstance(loss_fn, nn.CrossEntropyLoss)


@pytest.mark.advsecurenet
@pytest.mark.essential
def get_loss_function_invalid_string():
    with pytest.raises(ValueError, match="Unsupported loss function!"):
        get_loss_function("invalid_loss")
