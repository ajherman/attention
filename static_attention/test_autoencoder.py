import torch
import torch.nn as nn
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from utils import StaticMultiHeadAttention

# Test dataset
dataset = "MNIST"
if dataset == "MNIST":
    input_size = 784
    # Download MNIST
    test_dataset = MNIST(root='~/datasets', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
elif dataset == "CIFAR10":
    input_size = 3072
    # Download CIFAR10
    test_dataset = CIFAR10(root='~/datasets', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
model = StaticMultiHeadAttention(dm=input_size, dk=112, dv=112, N=1000, heads=7, dropout=0.2, rectify=0)

# Load saved weights
model.load_state_dict(torch.load('model_weights.pth'))

# Evaluate on test set
total_loss = 0
with torch.no_grad():
    for x, _ in test_loader:
        x = x.view(-1, input_size)
        y = model(x)
        loss = nn.MSELoss()(y, x)
        total_loss += loss.item() * x.size(0)

# Calculate average loss
average_loss = total_loss / len(test_dataset)

# Print average loss
print(f"Average Loss: {average_loss}")