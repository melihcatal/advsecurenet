from typing import List, Union

import torch
from torch import nn


class NormalizationLayer(nn.Module):
    """
    Normalization layer that normalizes the input tensor by subtracting the mean and dividing by the standard deviation.
    Each channel can have its own mean and standard deviation.
    """

    def __init__(self,
                 mean: Union[List[float], torch.Tensor],
                 std: Union[List[float], torch.Tensor]):
        """ 
        Constructor for the normalization layer.

        Args:
            mean (Union[List[float], torch.Tensor]): The mean values for each channel.
            std (Union[List[float], torch.Tensor]): The standard deviation values for each channel.
        """
        super().__init__()
        self.mean = self._convert_to_tensor(mean)
        self.std = self._convert_to_tensor(std)

    def forward(self, x):
        """
        Forward pass of the normalization layer.
        Assumes x is in shape [N, C, H, W].
        """
        return (x - self.mean) / self.std

    def _convert_to_tensor(self, x):
        """
        Converts the input to a tensor if it is not already.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x
