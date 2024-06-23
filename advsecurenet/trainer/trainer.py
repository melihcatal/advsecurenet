
import logging
import os
from typing import Union, cast

import click
import torch
from torch import nn, optim
from tqdm.auto import tqdm, trange

from advsecurenet.shared.optimizer import Optimizer
from advsecurenet.shared.scheduler import Scheduler
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.utils.loss import get_loss_function
from advsecurenet.utils.model_utils import save_model

logger = logging.getLogger(__name__)


class Trainer:
    """
    Base trainer module for training a model.
    """

    def __init__(self, config: TrainConfig):
        """
        Initialize the trainer.

        Args:
            config (TrainConfig): The train config.
        """
        self._config = config
        self._device = self._setup_device()
        self._model = self._setup_model()
        self._optimizer = self._setup_optimizer()
        self._loss_fn = get_loss_function(config.criterion)
        self._start_epoch = self._load_checkpoint_if_any()
        self._scheduler = self._setup_scheduler()

    def train(self) -> None:
        """
        Public method for training the model.
        """
        self._pre_training()
        for epoch in trange(self._start_epoch, self._config.epochs + 1, leave=True, position=0):
            self._run_epoch(epoch)
            if self._should_save_checkpoint(epoch):
                self._save_checkpoint(epoch, self._optimizer)
        self._post_training()

    def _setup_device(self) -> torch.device:
        """
        Setup the device.
        """
        device = self._config.processor or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _setup_model(self) -> torch.nn.Module:
        """
        Initializes the model and moves it to the device.
        """
        return self._config.model.to(self._device)

    def _setup_optimizer(self) -> optim.Optimizer:
        """
        Initializes the optimizer based on the given optimizer string or optim.Optimizer.

        Returns:
            optim.Optimizer: The optimizer. I.e. Adam, SGD, etc.
        """
        kwargs = self._config.optimizer_kwargs if self._config.optimizer_kwargs else {}
        optimizer = self._get_optimizer(
            self._config.optimizer,
            self._model,
            self._config.learning_rate,
            **kwargs
        )

        return optimizer

    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Initializes the scheduler based on the given scheduler string or torch.optim.lr_scheduler._LRScheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler. I.e. ReduceLROnPlateau, etc.
        """
        scheduler = self._get_scheduler(
            self._config.scheduler, self._optimizer)
        return scheduler

    def _get_scheduler(self, scheduler: Union[str, torch.optim.lr_scheduler._LRScheduler], optimizer: optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler based on the given scheduler string or torch.optim.lr_scheduler._LRScheduler.

        Args:
            scheduler (str or torch.optim.lr_scheduler._LRScheduler, optional): The scheduler. Defaults to None.
            optimizer (optim.Optimizer, optional): The optimizer. Required if scheduler is a string.   

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler. I.e. ReduceLROnPlateau, etc.
        """
        if scheduler is None:
            return None
        if isinstance(scheduler, str):
            if scheduler.upper() not in Scheduler.__members__:
                raise ValueError(
                    "Unsupported scheduler! Choose from: " + ", ".join([e.name for e in Scheduler]))
            scheduler_function_class = Scheduler[scheduler.upper()].value
            scheduler = scheduler_function_class(
                optimizer,
                **self._config.scheduler_kwargs if self._config.scheduler_kwargs else {}
            )
        elif not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            raise ValueError(
                "Scheduler must be a string or an instance of torch.optim.lr_scheduler._LRScheduler.")
        return cast(torch.optim.lr_scheduler._LRScheduler, scheduler)

    def _get_optimizer(self,
                       optimizer: Union[str, optim.Optimizer],
                       model: nn.Module,
                       learning_rate: float = 0.001,
                       **kwargs
                       ) -> optim.Optimizer:
        """
        Returns the optimizer based on the given optimizer string or optim.Optimizer.

        Args:
            optimizer (str or optim.Optimizer, optional): The optimizer. Defaults to Adam with learning rate 0.001.
            model (nn.Module, optional): The model to optimize. Required if optimizer is a string.
            learning_rate (float, optional): The learning rate. Defaults to 0.001.

        Returns:
            optim.Optimizer: The optimizer.

        Examples:

            >>> _get_optimizer("adam")
            >>> _get_optimizer(optim.Adam(model.parameters(), lr=0.001))

        """

        # if the optimizer is already an instance of optim.Optimizer, return it
        if isinstance(optimizer, optim.Optimizer):
            return optimizer

        if model is None and isinstance(optimizer, str):
            raise ValueError(
                "Model must be provided if optimizer is a string.")

        # if the model is provided but the optimizer not, initialize the default optimizer
        if model is not None and optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #  if the model is provided and the optimizer is a string, initialize the optimizer based on the string
        if model is not None and isinstance(optimizer, str):
            if optimizer.upper() not in Optimizer.__members__:
                raise ValueError(
                    "Unsupported optimizer! Choose from: " + ", ".join([e.name for e in Optimizer]))

            optimizer_class = Optimizer[optimizer.upper()].value
            optimizer = optimizer_class(
                model.parameters(),
                lr=learning_rate,
                **kwargs

            )

        return cast(optim.Optimizer, optimizer)

    def _load_checkpoint_if_any(self) -> int:
        """
        Loads the checkpoint if any and returns the start epoch.

        Returns:
            int: The start epoch.
        """
        try:
            start_epoch = 1
            if self._config.load_checkpoint and self._config.load_checkpoint_path:
                if os.path.isfile(self._config.load_checkpoint_path):
                    logger.info(
                        "Loading checkpoint from %s", self._config.load_checkpoint_path)
                    checkpoint = torch.load(self._config.load_checkpoint_path)
                    self._load_model_state_dict(checkpoint['model_state_dict'])
                    self._optimizer.load_state_dict(
                        checkpoint['optimizer_state_dict'])
                    self._assign_device_to_optimizer_state()
                    start_epoch = checkpoint['epoch'] + 1
                else:
                    logger.warning(
                        "Checkpoint file not found at %s", self._config.load_checkpoint_path)
            return start_epoch
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            return 1

    def _load_model_state_dict(self, state_dict):
        # Loads the given model state dict.
        self._model.load_state_dict(state_dict)

    def _get_model_state_dict(self) -> dict:
        # Returns the model state dict.
        return self._model.state_dict()

    def _assign_device_to_optimizer_state(self):
        # Default implementation
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)

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
            return f"{self._config.model._model_name}_{self._config.train_loader.dataset.__class__.__name__}_checkpoint"

    def _save_checkpoint(self, epoch: int, optimizer: optim.Optimizer) -> None:
        """
        Saves the checkpoint.

        Args:
            epoch (int): The current epoch.
            optimizer (optim.Optimizer): The optimizer.
        """
        checkpoint_sub_dir = "training"
        checkpoint_dir = self._config.save_checkpoint_path or os.path.join(
            os.getcwd(), f"checkpoints/{checkpoint_sub_dir}")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_checkpoint_prefix = self._get_save_checkpoint_prefix()
        checkpoint_filename = f"{save_checkpoint_prefix}_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self._get_model_state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        click.echo(click.style(
            f"Saved checkpoint to {checkpoint_path}", fg="green"))

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """
        Determines if a checkpoint should be saved based on the given epoch, the checkpoint interval and the current rank.
        Args:
            epoch (int): The current epoch.
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.  
        """
        return self._config.save_checkpoint and self._config.checkpoint_interval > 0 and epoch % self._config.checkpoint_interval == 0

    def _should_save_final_model(self) -> bool:
        """
        Determines if the final model should be saved based on the given save_final_model flag and the current rank.
        """
        return self._config.save_final_model

    def _save_final_model(self) -> None:
        """
        Saves the final model to the current directory with the name of the model variant and the dataset name.
        """
        if not self._config.save_model_path:
            self._config.save_model_path = os.getcwd()

        model_name = self._config.model._model_name if hasattr(
            self._config.model, "_model_name") else "model"
        dataset_name = self._config.train_loader.dataset.name if hasattr(
            self._config.train_loader.dataset, "name") else "dataset"

        if not self._config.save_model_name:
            self._config.save_model_name = f"{model_name}_{dataset_name}_final.pth"

        # if the same file exists, add a index to the file name
        index = 0
        while os.path.isfile(self._config.save_model_name):
            index += 1
            self._config.save_model_name = f"{model_name}_{dataset_name}_final_{index}.pth"

        save_model(
            model=self._model,
            filename=self._config.save_model_name,
            filepath=self._config.save_model_path,
            distributed=self._config.use_ddp
        )

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Runs the given batch.

        Args:
            source (torch.Tensor): The source.
            targets (torch.Tensor): The targets.

        Returns:
            float: The loss.
        """
        self._model.train()
        self._optimizer.zero_grad()
        output = self._model(source)
        loss = self._loss_fn(output, targets)
        loss.backward()
        self._optimizer.step()
        if self._scheduler:
            self._scheduler.step()
        return loss.item()

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs the given epoch.
        """
        total_loss = 0.0
        for _, (source, targets) in enumerate(tqdm(self._config.train_loader, leave=False)):
            source, targets = source.to(
                self._device), targets.to(self._device)
            loss = self._run_batch(source, targets)
            total_loss += loss

        total_loss /= len(self._config.train_loader)
        click.echo(
            click.style(f"Epoch {epoch} - Average loss: {total_loss:.4f}", fg="blue"))
        self._log_loss(epoch, total_loss)

    def _pre_training(self) -> None:
        # Method to run before training starts.
        self._model.train()

    def _post_training(self) -> None:
        # Method to run after training ends.
        if self._should_save_final_model():
            self._save_final_model()

    def _log_loss(self, epoch: int, loss: float, dir: str = None, filename: str = "loss.log") -> None:
        path = os.path.join(dir, filename) if dir else os.path.join(
            os.getcwd(), filename)
        # Save the loss to the log file. If the log file does not exist, create it in the current directory.
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("epoch,loss\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{loss}\n")
