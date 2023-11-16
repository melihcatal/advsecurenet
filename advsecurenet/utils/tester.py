import torch
from torch import nn
from tqdm import tqdm
from typing import cast, Sized, Union, Optional
from torch.utils.data import DataLoader


class Tester:
    """
    Tester class for testing a model on a given test_loader.

    Args:
        test_loader (DataLoader): The test loader.
        criterion (str or nn.Module, optional): The loss function. Defaults to nn.CrossEntropyLoss().
        device (torch.device, optional): The device to test on. Defaults to CPU.
    """

    def __init__(self, model: nn.Module, test_loader: DataLoader, criterion: Union[str, nn.Module] = torch.nn.CrossEntropyLoss(), device: torch.device = torch.device("cpu")) -> None:
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.loss_fn = self._get_loss_function(criterion)

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

    def test(self) -> tuple[float, float]:
        """
        Tests the model on the given test_loader. Prints the average loss and accuracy.

        Args:
            model (nn.Module): The model to test.
            test_loader (torch.utils.data.DataLoader): The test loader.
            criterion (nn.Module, optional): The loss function. Defaults to nn.CrossEntropyLoss().
            device (torch.device, optional): The device to test on. Defaults to CPU.

        Returns:
            tuple: A tuple containing the average loss and accuracy.

        """
        self.model.to(self.device)
        self.model.eval()
        test_loss: float = 0.0
        correct: int = 0
        print(f"Testing on {self.device}")
        with torch.no_grad():
            # Wrap the loop with tqdm for the progress bar
            for data, target in tqdm(self.test_loader, desc="Testing", unit="batch"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        dataset = cast(Sized, self.test_loader.dataset)
        test_loss /= len(dataset)

        accuracy = 100. * correct / len(dataset)
        print(
            f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)')

        return test_loss, accuracy
