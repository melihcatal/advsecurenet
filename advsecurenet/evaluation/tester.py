from typing import Sized, Union, cast

import torch
from torch import nn
from tqdm.auto import tqdm

from advsecurenet.shared.loss import Loss
from advsecurenet.shared.types.configs.test_config import TestConfig


class Tester:
    """
    Base tester module for testing a model.

    Args:
        config (TestConfig): The configuration for testing.
    """

    def __init__(self, config: TestConfig):
        self._model = config.model
        self._test_loader = config.test_loader
        self._device = config.processor
        self._loss_fn = self._get_loss_function(config.criterion)
        self._topk = int(config.topk)
        self._validate()

    def test(self) -> tuple[float, float]:
        """
        Tests the model on the given test_loader. Prints the average loss and accuracy.

        Returns:
            tuple: A tuple containing the average loss and accuracy.
        """
        self._model.to(self._device)
        self._model.eval()
        test_loss: float = 0.0
        correct_topk: int = 0
        with torch.no_grad():
            for data, target in tqdm(self._test_loader, desc="Testing", unit="batch"):
                data, target = data.to(self._device), target.to(self._device)
                output = self._model(data)
                test_loss += self._loss_fn(output, target).item()

                # Top-k accuracy
                _, pred_topk = output.topk(
                    self._topk, dim=1, largest=True, sorted=True)
                correct_topk += sum(target[i] in pred_topk[i]
                                    for i in range(target.size(0)))

        dataset = cast(Sized, self._test_loader.dataset)
        test_loss /= len(dataset)

        accuracy_topk = 100. * correct_topk / len(dataset)

        print(
            f'\nTest set: Average loss: {test_loss:.4f}, '
            f'Top-{self._topk} Accuracy: {correct_topk}/{len(self._test_loader.dataset)} ({accuracy_topk:.2f}%)')

        return test_loss, accuracy_topk

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

    def _validate(self) -> None:
        """ 
        Validates the configuration.

        Raises:
            ValueError: If the top-k value is less than 1 or greater than the number of classes.
        """
        # topk can't be less than 1 and greater than the number of classes
        if self._topk < 1:
            raise ValueError("Top-k value must be greater than 0.")
        if self._topk > self._model._num_classes:
            raise ValueError(
                "Top-k value must be less than or equal to the number of classes.")
