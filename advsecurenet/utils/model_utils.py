import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import torch
import os
import pkg_resources
from advsecurenet.shared.types import DeviceType

def train(model, train_loader, device=None, criterion=None, optimizer=None, epochs=3, learning_rate=0.001):
    """
    Trains the model on the given train_loader for the given number of epochs.

    Args:
        model (nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The train loader.
        device (torch.device, optional): The device to train on. Defaults to cuda if available, otherwise cpu.
        criterion (nn.Module, optional): The loss function. Defaults to nn.CrossEntropyLoss().
        optimizer (torch.optim, optional): The optimizer. Defaults to Adam with learning rate 0.001.
        epochs (int, optional): The number of epochs to train for. Defaults to 3.
        learning_rate (float, optional): The learning rate. Defaults to 0.001.

    """
    # First check if the model is pretrained
    if getattr(model, 'pretrained', False):
        raise ValueError("Cannot train a pretrained model!")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
        device (torch.device, optional): The device to test on. Defaults to cuda if available, otherwise cpu.

    Returns:
        tuple: A tuple containing the average loss and accuracy.

    """
 
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
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
    
    
    print(f"Loading model from {os.path.join(filepath, filename)}")
    
    model.load_state_dict(torch.load(os.path.join(filepath, filename), map_location=device.value))
    return model
