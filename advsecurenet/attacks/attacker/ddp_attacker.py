import os
import pickle

import click
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from advsecurenet.attacks.attacker import Attacker, AttackerConfig
from advsecurenet.dataloader.distributed_eval_sampler import \
    DistributedEvalSampler
from advsecurenet.distributed.ddp_base_task import DDPBaseTask


class DDPAttacker(DDPBaseTask, Attacker):
    """
    DDPAttacker module is specialized module for attacking a model using DistributedDataParallel in a multi-GPU setting.

    Args:
        config (AttackerConfig): The attacker config.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
    """

    def __init__(self, config: AttackerConfig, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size
        DDPBaseTask.__init__(self,
                             model=config.model,
                             rank=rank,
                             world_size=world_size
                             )
        config.dataloader.sampler = DistributedEvalSampler(
            config.dataloader.dataset, world_size, rank)
        Attacker.__init__(self, config)

    def execute(self):
        adv_images = self._execute_attack()
        if self._config.return_adversarial_images:
            self._store_results(adv_images)

    @staticmethod
    def gather_results(world_size) -> list:
        """
        Static method to gather results from all the processes. Each process stores the results in a temporary file. The results are gathered and returned.

        Args:
            world_size (int): The total number of processes.

        Returns:
            list: The gathered adversarial images.
        """
        gathered_adv_images = []
        for rank in range(world_size):
            output_path = f'./adv_images_{rank}.pkl'
            with open(output_path, 'rb') as f:
                batch_images = pickle.load(f)
                gathered_adv_images.extend(batch_images)
            os.remove(output_path)
        return gathered_adv_images

    def _store_results(self, adv_images):
        """
        Store results temporarily for gathering.
        """
        output_path = f'./adv_images_{self._rank}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(adv_images, f)

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
