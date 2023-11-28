
import os
from typing import Union, cast

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from advsecurenet.shared.loss import Loss
from advsecurenet.shared.optimizer import Optimizer
from advsecurenet.shared.types.configs.train_config import TrainConfig


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
        self.config = config
        self.device = self._setup_device()
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.loss_fn = self._get_loss_function(config.criterion)
        self.start_epoch = self._load_checkpoint_if_any()

    def _setup_device(self) -> torch.device:
        """
        Setup the device.
        """
        device = self.config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        return device

    def _setup_model(self) -> torch.nn.Module:
        """
        Initializes the model and moves it to the device.
        """
        return self.config.model.to(self.device)

    def _setup_optimizer(self) -> optim.Optimizer:
        """
        Initializes the optimizer based on the given optimizer string or optim.Optimizer.

        Returns:
            optim.Optimizer: The optimizer. I.e. Adam, SGD, etc.
        """
        optimizer = self._get_optimizer(
            self.config.optimizer, self.model, self.config.learning_rate)
        return optimizer

    def _get_loss_function(self, criterion: Union[str, nn.Module], **kwargs) -> nn.Module:
        """
        Returns the loss function based on the given loss_function string or nn.Module.

        Args:
            criterion (str or nn.Module, optional): The loss function. Defaults to nn.CrossEntropyLoss().

        Returns:
            nn.Module: The loss function.

        Examples:

            >>> _get_loss_function("cross_entropy")
            >>> _get_loss_function(nn.CrossEntropyLoss())

        """
        # If nothing is provided, use CrossEntropyLoss as default
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss(**kwargs)
        else:
            # if criterion is a string, convert it to the corresponding loss function
            if isinstance(criterion, str):

                if criterion.upper() not in Loss.__members__:
                    raise ValueError(
                        "Unsupported loss function! Choose from: " + ", ".join([e.name for e in Loss]))
                criterion_function_class = Loss[criterion.upper()].value
                criterion = criterion_function_class(**kwargs)
            elif not isinstance(criterion, nn.Module):
                raise ValueError(
                    "Criterion must be a string or an instance of nn.Module.")
        return cast(nn.Module, criterion)

    def _get_optimizer(self, optimizer: Union[str, optim.Optimizer], model: nn.Module, learning_rate: float = 0.001, **kwargs) -> optim.Optimizer:
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
        if model is None and isinstance(optimizer, str):
            raise ValueError(
                "Model must be provided if optimizer is a string.")

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            if isinstance(optimizer, str):
                if optimizer.upper() not in Optimizer.__members__:
                    raise ValueError(
                        "Unsupported optimizer! Choose from: " + ", ".join([e.name for e in Optimizer]))

                optimizer_class = Optimizer[optimizer.upper()].value
                optimizer = optimizer_class(
                    model.parameters(), lr=learning_rate, **kwargs)

            elif not isinstance(optimizer, optim.Optimizer):
                raise ValueError(
                    "Optimizer must be a string or an instance of optim.Optimizer.")
        return cast(optim.Optimizer, optimizer)

    def _load_checkpoint_if_any(self) -> int:
        """
        Loads the checkpoint if any and returns the start epoch.

        Returns:
            int: The start epoch.
        """
        start_epoch = 1
        if self.config.load_checkpoint and self.config.load_checkpoint_path:
            if os.path.isfile(self.config.load_checkpoint_path):
                print(
                    f"=> loading checkpoint '{self.config.load_checkpoint_path}'")
                checkpoint = torch.load(self.config.load_checkpoint_path)
                start_epoch = checkpoint['epoch']
                self._load_model_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])
                self._assign_device_to_optimizer_state()
                start_epoch = checkpoint['epoch'] + 1
            else:
                print(
                    f"=> no checkpoint found at '{self.config.load_checkpoint_path}'")
        return start_epoch

    def _load_model_state_dict(self, state_dict):
        # Loads the given model state dict.
        self.model.load_state_dict(state_dict)

    def _get_model_state_dict(self) -> dict:
        # Returns the model state dict.
        return self.model.state_dict()

    def _assign_device_to_optimizer_state(self):
        # Default implementation
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def _get_save_checkpoint_prefix(self) -> str:
        """
        Returns the save checkpoint prefix.

        Returns:
            str: The save checkpoint prefix.

        Notes:
            If the save checkpoint name is provided, it will be used as the prefix. Otherwise, the model variant and the dataset name will be used as the prefix.
        """

        if self.config.save_checkpoint_name:
            return self.config.save_checkpoint_name
        else:
            return f"{self.config.model.model_name}_{self.config.train_loader.dataset.__class__.__name__}_checkpoint"

    def _save_checkpoint(self, epoch: int, optimizer: optim.Optimizer) -> None:
        """
        Saves the checkpoint.

        Args:
            epoch (int): The current epoch.
            optimizer (optim.Optimizer): The optimizer.
        """
        checkpoint_sub_dir = "training"
        checkpoint_dir = self.config.save_checkpoint_path or os.path.join(
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
        print(f"=> saved checkpoint '{checkpoint_path}'")

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """
        Determines if a checkpoint should be saved based on the given epoch, the checkpoint interval and the current rank.
        Args:
            epoch (int): The current epoch.
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.  
        """
        return self.config.save_checkpoint and self.config.checkpoint_interval > 0 and epoch % self.config.checkpoint_interval == 0

    def _should_save_final_model(self) -> bool:
        """
        Determines if the final model should be saved based on the given save_final_model flag and the current rank.
        """
        return self.config.save_final_model

    def _save_final_model(self) -> None:
        """
        Saves the final model to the current directory with the name of the model variant and the dataset name.
        """
        file_name = f"{self.config.model.model_name}_{self.config.train_loader.dataset.__class__.__name__}_final.pth"
        # if the same file exists, add a index to the file name
        index = 0
        while os.path.isfile(file_name):
            index += 1
            file_name = f"{self.config.model.model_name}_{self.config.train_loader.dataset.__class__.__name__}_final_{index}.pth"

        torch.save(self._get_model_state_dict(), file_name)
        print(f"=> saved final model '{file_name}'")

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Runs the given batch.

        Args:
            source (torch.Tensor): The source.
            targets (torch.Tensor): The targets.

        Returns:
            float: The loss.
        """
        # Set the model to train mode
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs the given epoch.
        """
        total_loss = 0.0

        # for source, targets in self.config.train_loader:
        for batch_idx, (source, targets) in enumerate(tqdm(self.config.train_loader)):
            source, targets = source.to(
                self.device), targets.to(self.device)
            loss = self._run_batch(source, targets)
            total_loss += loss

        total_loss /= len(self.config.train_loader)
        print(f"Epoch {epoch} loss: {total_loss}")

    def _pre_training(self) -> None:
        # Method to run before training starts.
        self.model.train()

    def _post_training(self) -> None:
        # Method to run after training ends.
        if self._should_save_final_model():
            self._save_final_model()

    def train(self) -> None:
        """
        Public method for training the model.
        """
        self._pre_training()
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            self._run_epoch(epoch)
            if self._should_save_checkpoint(epoch):
                self._save_checkpoint(epoch, self.optimizer)
        self._post_training()
