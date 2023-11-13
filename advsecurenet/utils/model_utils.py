import torch
import torch.optim as optim
import os
import pkg_resources
import requests
from typing import Optional, cast, Any, Union, cast
from torch import nn
from tqdm import tqdm
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.shared.loss import Loss
from advsecurenet.shared.optimizer import Optimizer


def _get_loss_function(criterion: Union[str, nn.Module], **kwargs) -> nn.Module:
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


def _get_optimizer(optimizer: Union[str, optim.Optimizer], model: nn.Module, learning_rate: float = 0.001, **kwargs) -> optim.Optimizer:
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
        raise ValueError("Model must be provided if optimizer is a string.")

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


def _setup_device(config: Union[TrainConfig, AdversarialTrainingConfig]) -> torch.device:
    return config.device if config.device else torch.device("cpu")


# Replace Any with the actual type
def _initialize_optimizer(config:  Union[TrainConfig, AdversarialTrainingConfig]) -> optim.Optimizer:
    return _get_optimizer(config.optimizer, config.model, config.learning_rate)


def _load_checkpoint_if_any(config: Union[TrainConfig, AdversarialTrainingConfig], device: torch.device, optimizer: optim.Optimizer) -> int:
    start_epoch = 1
    if config.load_checkpoint and config.load_checkpoint_path:
        if os.path.isfile(config.load_checkpoint_path):
            print(
                f"Loading checkpoint from '{config.load_checkpoint_path}'")
            checkpoint = torch.load(
                config.load_checkpoint_path, map_location=device)
            config.model.load_state_dict(checkpoint['model_state_dict'])
            config.model.to(device)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(
                f"No checkpoint found at '{config.load_checkpoint_path}', starting from scratch.")
    return start_epoch


def _get_save_checkpoint_prefix(config: Union[TrainConfig, AdversarialTrainingConfig]) -> str:
    if config.save_checkpoint_name:
        return config.save_checkpoint_name
    else:
        return f"{config.model.model_variant}_{config.train_loader.dataset.__class__.__name__}_checkpoint"


def _save_checkpoint(config: Union[TrainConfig, AdversarialTrainingConfig], epoch: int, optimizer: optim.Optimizer) -> None:
    checkpoint_sub_dir = "adversarial_training" if isinstance(
        config, AdversarialTrainingConfig) else "training"
    # if save_checkpoint_path is not provided, save in the current working directory
    checkpoint_dir = config.save_checkpoint_path or os.path.join(
        os.getcwd(), f"checkpoints/{checkpoint_sub_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_checkpoint_prefix = _get_save_checkpoint_prefix(config)
    checkpoint_filename = f"{save_checkpoint_prefix}_{epoch}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': config.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at '{checkpoint_path}'")


def _train_model(train_config: TrainConfig, device: torch.device, optimizer: optim.Optimizer) -> None:
    train_config.model.to(device)
    train_config.model.train()
    loss_function = _get_loss_function(train_config.criterion)
    start_epoch = _load_checkpoint_if_any(train_config, device, optimizer)
    for epoch in range(start_epoch, train_config.epochs + 1):
        total_loss = _run_training_epoch(
            train_config, device, optimizer, loss_function, epoch)
        average_loss = total_loss / len(train_config.train_loader)
        print(f'Epoch {epoch} - Average Loss: {average_loss:.6f}')
        if train_config.save_checkpoint and epoch % train_config.checkpoint_interval == 0:
            _save_checkpoint(train_config, epoch, optimizer=optimizer)
    print("Training completed.")


def _run_training_epoch(train_config: TrainConfig, device: torch.device, optimizer: Any, loss_function: nn.Module, epoch: int) -> float:
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(tqdm(train_config.train_loader, desc=f"Epoch {epoch}/{train_config.epochs}", total=len(train_config.train_loader))):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = train_config.model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def train(train_config: TrainConfig) -> None:
    """
    Trains the model based on the given train_config.

    Args:
        train_config (TrainConfig): The train configuration.
    """
    if getattr(train_config.model, 'pretrained', False):
        raise ValueError("Cannot train a pretrained model!")
    device = _setup_device(train_config)
    optimizer = _initialize_optimizer(train_config)
    print(f"Training on {device}")
    _train_model(train_config, device, optimizer)


def test(model: nn.Module,
         test_loader: torch.utils.data.DataLoader,
         criterion: str or nn.Module = None,
         device: torch.device = torch.device("cpu")) -> None:
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

    loss_function = _get_loss_function(criterion)
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    print(f"Testing on {device}")
    with torch.no_grad():
        # Wrap the loop with tqdm for the progress bar
        for data, target in tqdm(test_loader, desc="Testing", unit="batch"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy


def save_model(model: nn.Module,
               filename: str,
               filepath: str = None):
    """
    Saves the model weights to the given filepath.

    Args:
        model (nn.Module): The model to save.
        filename (str): The filename to save the model weights to.
        filepath (str, optional): The filepath to save the model weights to. Defaults to weights directory.

    """

    if filepath is None:
        filepath = pkg_resources.resource_filename("advsecurenet", "weights")

    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # add .pth extension if not present
    if not filename.endswith(".pth"):
        filename = filename + ".pth"

    torch.save(model.state_dict(), os.path.join(filepath, filename))


def load_model(model, filename, filepath=None, device: torch.device = torch.device("cpu")):
    """
    Loads the model weights from the given filepath.

    Args:
        model (nn.Module): The model to load the weights into.
        filename (str): The filename to load the model weights from.
        filepath (str, optional): The filepath to load the model weights from. Defaults to weights directory.
        device (torch.device, optional): The device to load the model weights to. Defaults to CPU.
    """
    # check if filename contains a path if so, use that instead of filepath
    if os.path.dirname(filename):
        filepath = os.path.dirname(filename)
        filename = os.path.basename(filename)

    if filepath is None:
        filepath = pkg_resources.resource_filename("advsecurenet", "weights")

    # add .pth extension if not present
    if not filename.endswith(".pth"):
        filename = filename + ".pth"

    model.load_state_dict(torch.load(os.path.join(
        filepath, filename), map_location=device))
    return model


def download_weights(model_name: Optional[str] = None,
                     dataset_name: Optional[str] = None,
                     filename: Optional[str] = None,
                     save_path: Optional[str] = None):
    """
    Downloads model weights from a remote source based on the model and dataset names.

    Args:
        model_name (str): The name of the model (e.g. "resnet50").
        dataset_name (str): The name of the dataset the model was trained on (e.g. "cifar10").
        filename (str, optional): The filename of the weights on the remote server. If provided, this will be used directly.
        save_path (str, optional): The directory to save the weights to. Defaults to weights directory.

    Examples:

        >>> download_weights(model_name="resnet50", dataset_name="cifar10")
        Downloaded weights to /home/user/advsecurenet/weights/resnet50_cifar10.pth

        >>> download_weights(filename="resnet50_cifar10.pth")
        Downloaded weights to /home/user/advsecurenet/weights/resnet50_cifar10.pth
    """

    base_url = "https://advsecurenet.s3.eu-central-1.amazonaws.com/weights/"

    # Generate filename and remote_url based on model_name and dataset_name if filename is not provided
    if not filename:
        if model_name is None or dataset_name is None:
            raise ValueError(
                "Both model_name and dataset_name must be provided if filename is not specified.")
        filename = f"{model_name}_{dataset_name}_weights.pth"

    remote_url = os.path.join(base_url, filename)

    if save_path is None:
        save_path = pkg_resources.resource_filename("advsecurenet", "weights")

    # Ensure directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    local_file_path = os.path.join(save_path, filename)

    # Only download if file doesn't exist
    if not os.path.exists(local_file_path):
        progress_bar = None
        try:
            response = requests.get(remote_url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            progress_bar = tqdm(total=total_size, unit='B',
                                unit_scale=True, desc=filename)

            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    progress_bar.update(len(chunk))

            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                raise Exception(
                    "Error, something went wrong while downloading.")

            print(f"Downloaded weights to {local_file_path}")
        except (Exception, KeyboardInterrupt) as e:
            # If any error occurs, delete the file and re-raise the exception
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            if progress_bar is not None:
                progress_bar.close()
            raise e

    else:
        print(f"File {local_file_path} already exists. Skipping download.")

    return local_file_path
