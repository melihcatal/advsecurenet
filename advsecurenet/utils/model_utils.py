import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import torch
import os

def train(model, train_loader, device=None, criterion=None, optimizer=None, epochs=3, learning_rate=0.001):
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


def save_model(model, filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    return model
