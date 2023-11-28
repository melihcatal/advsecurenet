import os
import torch
import pkg_resources
import requests
from torch import nn
from tqdm import tqdm
from typing import Optional


def save_model(model: nn.Module,
               filename: str,
               filepath: Optional[str] = None,
               distributed: bool = False,
               ):
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

    if distributed:
        torch.save(model.module.state_dict(), os.path.join(filepath, filename))
    else:
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
