import torch
import torch.nn as nn
from utils import StaticAttentionHead
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from utils import *

dataset = "CIFAR10"

if dataset == "MNIST":
    input_size = 784
    # Download MNIST
    train_dataset = MNIST(root='~/datasets', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='~/datasets', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
elif dataset == "CIFAR10":
    input_size = 3072
    # Download CIFAR10
    train_dataset = CIFAR10(root='~/datasets', train=True, download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='~/datasets', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define static transformer
if dataset == "MNIST":
    model = StaticTransformer(dm=input_size, dk=112, dv=112, N=1000, heads=7, layers=1, n_classes=10)
elif dataset == "CIFAR10":
    model = StaticTransformer(dm=input_size, dk=3*128, dv=3*128, N=1000, heads=8, layers=1, n_classes=10)
# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# Train
for epoch in range(10):
    for itr, (x, y) in enumerate(train_loader):
        x = x.view(-1, input_size)
        y_pred = model(x)
        loss = model.loss(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            predicted = torch.argmax(y_pred, dim=1)
            correct = (predicted == y).sum().item()
            accuracy = correct / y.size(0)

        # Print loss and accuracy after each batch
        if itr % 50 == 0:
            print(f'Epoch {epoch}, Batch Loss: {loss.item()}, Accuracy: {accuracy}')

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for x,y in test_loader:
            x=x.view(-1, input_size)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    # Print accuracy after each epoch
    print(f'Epoch {epoch}, Accuracy: {correct/total}')    
