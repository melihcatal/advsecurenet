import os

import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group

from advsecurenet.utils.network import find_free_port


class DDPCoordinator:
    """
    The generic DDP DDPTrainer class. This class is used to train a model using DistributedDataParallel. 
    """

    def __init__(self, ddp_func, world_size, *args, **kwargs):
        """
        Initialize the  DDP DDPTrainer. DDPCoordinator is a wrapper class for DistributedDataParallel used for multi-GPU training.
        The module can be used for both adversarial and non-adversarial training.

        Args:
            ddp_func: The ddp function to be called by each process.
            world_size: The number of processes to spawn.
            *args: The arguments to be passed to the training function.
            **kwargs: The keyword arguments to be passed to the training function.

        Note:
            Currently, the module uses single-node multi-GPU training. The module uses the nccl backend by default.
            It finds a free port on the machine and uses it as the master port.


        """
        self.ddp_func = ddp_func
        self.world_size = world_size
        self.args = args
        self.kwargs = kwargs
        self.port = find_free_port()
        self.backend = 'nccl'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self.port)

    def ddp_setup(self, rank: int):
        """
        DDP setup function. This function is called by each process to setup the DDP environment.
        Sets the master address and port and initializes the process group.
        Automatically finds a free port on the machine and uses it as the master port.

        The default backend is nccl.
        """
        init_process_group(backend=self.backend,
                           rank=rank,
                           world_size=self.world_size)

    def run_process(self, rank: int):
        """
        Setup DDP and call the training function.
        """
        self.ddp_setup(rank)
        self.ddp_func(rank, self.world_size, *self.args, **self.kwargs)
        destroy_process_group()

    def run(self):
        """
        Spawn the processes for DDP training.
        """
        mp.spawn(self.run_process, nprocs=self.world_size, join=True)
