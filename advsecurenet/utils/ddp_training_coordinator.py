import os
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from advsecurenet.utils.network import find_free_port


class DDPTrainingCoordinator:
    """
    The generic DDP DDPTrainer class. This class is used to train a model using DistributedDataParallel. 
    """

    def __init__(self, train_func, world_size, *args, **kwargs):
        """
        Initialize the  DDP DDPTrainer. DDPTrainingCoordinator is a wrapper class for DistributedDataParallel used for multi-GPU training.
        The module can be used for both adversarial and non-adversarial training.

        Args:
            train_func: The training function to be called by each process.
            world_size: The number of processes to spawn.
            *args: The arguments to be passed to the training function.
            **kwargs: The keyword arguments to be passed to the training function.

        Note:
            Currently, the trainer only supports training on a single machine with multiple GPUs.
            It assumes that the port 12355 is free on the machine.

        Example:
            >>> from advsecurenet.utils.ddp_training_coordinator import DDPTrainingCoordinator
            >>> def main_training_function(rank, model, dataset, train_config, attack_config):
            >>>     # Define the training logic here
            >>>     pass
            >>> trainer = GenericDDPTrainingCoordinator(main_training_function, world_size=3, model=model,
                            dataset=dataset, train_config=train_config, attack_config=attack_config)
            >>> trainer.run()

        """
        self.train_func = train_func
        self.world_size = world_size
        self.args = args
        self.kwargs = kwargs
        self.port = find_free_port()
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(self.port)

    def ddp_setup(self, rank: int):
        """
        DDP setup function. This function is called by each process to setup the DDP environment.
        Sets the master address and port and initializes the process group.
        Automatically finds a free port on the machine and uses it as the master port.

        The default backend is nccl.
        """
        init_process_group(backend='nccl', rank=rank,
                           world_size=self.world_size)

    def run_process(self, rank: int):
        """
        Setup DDP and call the training function.
        """
        self.ddp_setup(rank)
        self.train_func(rank, self.world_size, *self.args, **self.kwargs)
        destroy_process_group()

    def run(self):
        """
        Spawn the processes for DDP training.
        """
        mp.spawn(self.run_process, args=(),
                 nprocs=self.world_size, join=True)
