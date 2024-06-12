import os
import pickle

import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from advsecurenet.attacks.attacker import Attacker, AttackerConfig
from advsecurenet.dataloader.distributed_eval_sampler import \
    DistributedEvalSampler
from advsecurenet.utils.ddp import set_visible_gpus
from advsecurenet.utils.network import find_free_port


class DDPAttacker(Attacker):
    """
    DDPAttacker is a specialized module for attacking a model using DistributedDataParallel in a multi-GPU setting.
    """

    def __init__(self, config: AttackerConfig, gpu_ids: list, **kwargs) -> None:
        """
        Initializes the DDPAttacker object.

        Args:
            config (AttackerConfig): The configuration object for the attacker.
            gpu_ids (list): A list of GPU IDs to be used for distributed training.
        """
        self._world_size = len(gpu_ids)
        self._gpu_ids = gpu_ids
        self._rank = 0
        set_visible_gpus(self._gpu_ids)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(find_free_port())
        torch.multiprocessing.set_sharing_strategy('file_system')
        self._kwargs = kwargs
        super().__init__(config, **kwargs)

    def execute(self) -> list:
        """ 
        Executes the attack.
        """
        if self._gpu_ids is None or len(self._gpu_ids) == 0:
            self._gpu_ids = list(range(torch.cuda.device_count()))

        world_size = len(self._gpu_ids)

        mp.spawn(self._ddp_execute, nprocs=world_size, join=True)

        # Only the main process needs to gather the results
        if self._rank == 0 and self._config.return_adversarial_images:
            gathered_adv_images = self._gather_results()
            adversarial_images = [
                img for sublist in gathered_adv_images for img in sublist]
            return adversarial_images
        else:
            return []

    def _ddp_execute(self, rank: int):
        """
        Executes the attack.

        Args:
            rank (int): The rank of the current process in the distributed training.
        """
        self._setup(rank)
        adv_images = self._execute_attack()
        if self._config.return_adversarial_images:
            self._store_results(adv_images)
        self._cleanup()

    def _store_results(self, adv_images):
        """
        Store results temporarily for gathering.
        """
        output_path = f'./adv_images_{self._rank}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(adv_images, f)

    def _gather_results(self):
        """
        Gather and flatten results from all processes.
        """
        gathered_adv_images = []
        for rank in range(self._world_size):
            output_path = f'./adv_images_{rank}.pkl'
            with open(output_path, 'rb') as f:
                batch_images = pickle.load(f)
                # Flatten the batches into a single list
                gathered_adv_images.extend(batch_images)
            # Remove the temporary file
            os.remove(output_path)
        return gathered_adv_images

    def _setup(self, rank: int) -> None:
        """
        Sets up the distributed training environment.

        Args:
            rank (int): The rank of the current process in the distributed training.
        """
        dist.init_process_group(backend='nccl',
                                rank=rank,
                                world_size=self._world_size)
        device = torch.device(f'cuda:{rank}')

        self._rank = rank
        # Set the device to the correct GPU
        self._device = device

        # set cuda device
        torch.cuda.set_device(device)

        model = self._config.model.to(device)
        model = DDP(model, device_ids=[device])
        self._model = model

        # update the sampler to be a distributed sampler
        sampler = DistributedEvalSampler(self._config.dataloader.dataset)
        self._config.dataloader.sampler = sampler

        self._dataloader = self._create_dataloader()

    def _cleanup(self) -> None:
        """
        Cleans up the distributed training environment.
        """
        dist.destroy_process_group()
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    def _get_iterator(self) -> iter:
        """
        Returns an iterator for the dataloader. If the rank is 0, it returns a tqdm iterator.

        Returns:
            iter: An iterator for the dataloader.
        """
        if self._rank == 0:
            # Only initialize tqdm in the master process
            return tqdm(self._dataloader, leave=False, position=1, unit="batch", desc="Generating adversarial samples", colour="red")
        else:
            return self._dataloader

    def _summarize_metric(self, name, value):
        local_results = torch.tensor(value, device=self._device)
        dist.all_reduce(local_results, op=dist.ReduceOp.AVG)
        if self._rank == 0:
            click.secho(
                f"{name.replace('_', ' ').title()}: {local_results.item():.4f}", fg='green')
