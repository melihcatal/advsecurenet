import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from advsecurenet.distributed.ddp_base_task import DDPBaseTask
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.trainer.trainer import Trainer


class DDPTrainer(DDPBaseTask, Trainer):
    """
    DDPTrainer module is specialized module for training a model using DistributedDataParallel in a multi-GPU setting.

    Args:
        config (TrainConfig): The train config.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Examples:

            >>> trainer = DDPTrainer(config, rank, world_size)
            >>> trainer.train()

    """

    def __init__(self, config: TrainConfig, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size
        DDPBaseTask.__init__(
            self,
            model=config.model,
            rank=rank,
            world_size=world_size
        )
        Trainer.__init__(self, config)

    def _load_model_state_dict(self, state_dict):
        """
        Loads the given model state dict.
        """
        self._model.module.load_state_dict(state_dict)

    def _get_model_state_dict(self) -> dict:
        # Returns the model state dict.
        return self._model.module.state_dict()

    def _assign_device_to_optimizer_state(self):
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(self._rank)

    def _get_save_checkpoint_prefix(self) -> str:
        """
        Returns the save checkpoint prefix.

        Returns:
            str: The save checkpoint prefix.

        Notes:
            If the save checkpoint name is provided, it will be used as the prefix. Otherwise, the model variant and the dataset name will be used as the prefix.
        """

        if self._config.save_checkpoint_name:
            return self._config.save_checkpoint_name
        else:
            return f"{self._config.model.model_name}_{self._config.train_loader.dataset.__class__.__name__}_checkpoint"

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """
        Determines if a checkpoint should be saved based on the given epoch, the checkpoint interval and the current rank.
        Args:
            epoch (int): The current epoch.
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.  
        """
        return self._rank == 0 and self._config.save_checkpoint and self._config.checkpoint_interval > 0 and epoch % self._config.checkpoint_interval == 0

    def _should_save_final_model(self) -> bool:
        """
        Determines if the final model should be saved based on the given save_final_model flag and the current rank.
        """
        return self._rank == 0 and self._config.save_final_model

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs the given epoch.
        Args:
            epoch (int): Current epoch number.
        """
        total_loss = 0.0
        sampler = self._config.train_loader.sampler
        assert isinstance(
            sampler, DistributedSampler), "Sampler must be of type DistributedSampler"
        sampler.set_epoch(epoch)

        if self._rank == 0:
            # Only initialize tqdm in the master process
            data_iterator = tqdm(self._config.train_loader,
                                 leave=False, position=1)
        else:
            data_iterator = self._config.train_loader
        for source, targets in data_iterator:
            source, targets = source.to(
                self._device), targets.to(self._device)
            loss = self._run_batch(source, targets)
            total_loss += loss

        # Compute average loss across all batches and all processes
        total_loss /= len(self._config.train_loader) * self._world_size

        if self._rank == 0:
            self._log_loss(epoch, total_loss)
