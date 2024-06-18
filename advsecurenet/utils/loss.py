from typing import Union, cast

import torch
from torch import nn

from advsecurenet.shared.loss import Loss


def get_loss_function(criterion: Union[str, nn.Module], **kwargs) -> nn.Module:
    """
    Returns the loss function based on the given loss_function string or nn.Module.

    Args:
        criterion (str or nn.Module, optional): The loss function. Defaults to nn.CrossEntropyLoss().

    Returns:
        nn.Module: The loss function.

    Examples:

        >>> get_loss_function("cross_entropy")
        >>> get_loss_function(nn.CrossEntropyLoss())

    """
    # If nothing is provided, use CrossEntropyLoss as default
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(**kwargs)
    else:
        # if criterion is a string, convert it to the corresponding loss function
        if isinstance(criterion, str):
            if criterion.upper() not in Loss.__members__:
                raise ValueError(
                    "Unsupported loss function! Choose from: " + ", ".join([e.name for e in Loss]))
            criterion_function_class = Loss[criterion.upper()].value
            criterion = criterion_function_class(**kwargs)
        elif not isinstance(criterion, nn.Module):
            raise ValueError(
                "Criterion must be a string or an instance of nn.Module.")
    return cast(nn.Module, criterion)
