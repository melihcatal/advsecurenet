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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the normalization layer.
        Assumes x is in shape [N, C, H, W].

        Args: 
            x (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor.

        Note:
            N: Batch size
            C: Number of channels
            H: Height of the image
            W: Width of the image

        """
        input_device = x.device
        mean = self.mean.to(input_device)
        std = self.std.to(input_device)

        return (x - mean) / std

    def _convert_to_tensor(self, x):
        """
        Converts the input to a tensor if it is not already and reshapes it to [1, C, 1, 1] for broadcasting.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.float()  # Ensure the tensor is of type float for division operations
        # Reshape to [1, C, 1, 1] to broadcast along channel dimension
        return x.view(1, -1, 1, 1)
