import torch
from typing import Union


class DeviceManager:
    """
    Device manager module for handling device placement in both single and distributed modes. In single mode, the device manager will place the tensors on the specified device. In distributed mode, the device manager will assume that the tensors are already placed correctly.
    This centralizes the device placement logic and makes it easier to switch between single and distributed modes.
    """

    def __init__(self, device: Union[str, torch.device], distributed_mode: bool):
        self.initial_device = device
        self.distributed_mode = distributed_mode

    def get_current_device(self):
        """
        Returns the current device. In distributed mode, it returns the device of the current process.
        In single mode, it returns the initialized device.
        """
        if self.distributed_mode:
            return torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            return self.initial_device

    def to_device(self, *args):
        """
        Places the provided tensors on the correct device based on the current mode.
        In distributed mode, it places tensors on the device of the current process.
        In single mode, it places tensors on the initialized device.
        """
        device = self.get_current_device()
        processed_args = [
            arg.to(device) if arg is not None else None for arg in args]
        return processed_args[0] if len(processed_args) == 1 else processed_args
