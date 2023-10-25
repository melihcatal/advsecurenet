import torch
import torch.optim as optim
import os
import pkg_resources
import requests
from torch import nn
from tqdm import tqdm
from advsecurenet.shared.types import DeviceType
from advsecurenet.utils.get_device import get_device

def train(model, train_loader, device=None, criterion=None, optimizer=None, epochs=3, learning_rate=0.001):
    """
    Trains the model on the given train_loader for the given number of epochs.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The train loader.
        device (DeviceType, optional): The device to train on. Defaults to CPU
        criterion (nn.Module, optional): The loss function. Defaults to nn.CrossEntropyLoss().
        optimizer (torch.optim, optional): The optimizer. Defaults to Adam with learning rate 0.001.
        epochs (int, optional): The number of epochs to train for. Defaults to 3.
        learning_rate (float, optional): The learning rate. Defaults to 0.001.

    """
    # First check if the model is pretrained
    if getattr(model, 'pretrained', False):
        raise ValueError("Cannot train a pretrained model!")

    if device is None:
        device = get_device().value
    
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on {device}")
    model.to(device)
    model.train()  # Set the model to training mode

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} ", total=len(train_loader))):
            data, target = data.to(device), target.to(device)  # Transfer data and target to device
            
            optimizer.zero_grad() 
            outputs = model(data)  
            loss = criterion(outputs, target)  
            loss.backward()  
            optimizer.step() 
            
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} - Average Loss: {average_loss:.6f}')




def test(model, test_loader, criterion=None, device=None):
    """
    Tests the model on the given test_loader. Prints the average loss and accuracy.

    Args:
        model (nn.Module): The model to test.
        test_loader (torch.utils.data.DataLoader): The test loader.
        criterion (nn.Module, optional): The loss function. Defaults to nn.CrossEntropyLoss().
        device (DeviceType, optional): The device to test on. Defaults to CPU.

    Returns:
        tuple: A tuple containing the average loss and accuracy.

    """
    if device is None:
        device = get_device().value
    
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

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
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy


def save_model(model, filename, filepath = None):
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

def load_model(model, filename, filepath= None, device=None):
    """
    Loads the model weights from the given filepath.

    Args:
        model (nn.Module): The model to load the weights into.
        filename (str): The filename to load the model weights from.
        filepath (str, optional): The filepath to load the model weights from. Defaults to weights directory.
        device (DeviceType, optional): The device to load the model weights to. Defaults to CPU.
    """

    if filepath is None:
        filepath = pkg_resources.resource_filename("advsecurenet", "weights")

    # add .pth extension if not present
    if not filename.endswith(".pth"):
        filename = filename + ".pth"
    
    if device is None:
        device = DeviceType.CPU
    
    if isinstance(device, DeviceType):
        device = device.value
        
    model.load_state_dict(torch.load(os.path.join(filepath, filename), map_location=device))
    return model


def download_weights(model_name=None, dataset_name=None, filename=None, save_path=None):
    """
    Downloads model weights from a remote source based on the model and dataset names.

    Args:
        model_name (str, optional): The name of the model (e.g. "resnet50").
        dataset_name (str, optional): The name of the dataset the model was trained on (e.g. "cifar10").
        filename (str, optional): The filename of the weights on the remote server. If provided, this will be used directly.
        save_path (str, optional): The directory to save the weights to. Defaults to weights directory.

    Examples:

        >>> download_weights(model_name="resnet50", dataset_name="cifar10")
        Downloaded weights to /home/user/advsecurenet/weights/resnet50_cifar10.pth

        >>> download_weights(filename="resnet50_cifar10.pth")
        Downloaded weights to /home/user/advsecurenet/weights/resnet50_cifar10.pth
    """
    
    base_url = "https://advsecurenet.s3.eu-central-1.amazonaws.com/weights/"  # Replace with the base URL of your remote server
    
    # Generate filename and remote_url based on model_name and dataset_name if filename is not provided
    if not filename:
        if model_name is None or dataset_name is None:
            raise ValueError("Both model_name and dataset_name must be provided if filename is not specified.")
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
            
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename)
            
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            
            progress_bar.close()
            
            if total_size != 0 and progress_bar.n != total_size:
                raise Exception("Error, something went wrong while downloading.")
            
            print(f"Downloaded weights to {local_file_path}")
        except (Exception, KeyboardInterrupt) as e:
            # If any error occurs, delete the file and re-raise the exception
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            if progress_bar is not None:
                progress_bar.close()
            raise e
        
    else:
        raise FileExistsError(f"Weights already exist at {local_file_path}")
    
    return local_file_path
