import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from advsecurenet.models.base_model import BaseModel


class DDPBaseTask:
    """ 
    Base class for DistributedDataParallel tasks.

    Args:
        model (BaseModel): The model to be used.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """

    def __init__(self, model: BaseModel, rank: int, world_size: int):
        self._model = model
        self._rank = rank
        self._world_size = world_size

    def _setup_device(self) -> torch.device:
        """
        Initializes the device based on the rank of the current process.

        Returns:   
            torch.device: The device.
        """
        torch.cuda.set_device(self._rank)
        return torch.device(f"cuda:{self._rank}")

    def _setup_model(self) -> torch.nn.parallel.DistributedDataParallel:
        """
        Initializes the model based on the rank of the current process.

        Returns:
            DistributedDataParallel: The model.
        """
        model = self._model.to(self._rank)
        return DDP(model, device_ids=[self._rank])
