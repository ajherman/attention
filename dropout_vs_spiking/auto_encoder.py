import torch
import argparse
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from utils import *
import numpy as np
import matplotlib.pyplot as plt

# Create matrix for low rank random data generation
np.random.seed(1234)

# Define the neural network architecture

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
parser.add_argument('--data-dir', type=str, default='~/datasets', help='Directory to store data')
parser.add_argument('--csv-file', type=str, default='data.csv', help='CSV file location')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--mom', type=float, default=0.5, help='Momentum')
parser.add_argument('--wd',type=float,default=0.001, help='Weight decay')
parser.add_argument('--dr',type=float,default=0.0,help='Dropout rate')
parser.add_argument('--normalize-weights',action='store_true',help='Normalize the weights to unit norm')
parser.add_argument('--save-dir',type=str,default='./results',help='Save directory')
parser.add_argument('--tied',)
args = parser.parse_args()

dataset_name = args.dataset
device = args.device
data_dir = args.data_dir
csv_file = args.csv_file
num_epochs = args.epochs
learning_rate = args.lr
momentum = args.mom
weight_decay=args.wd 
dropout_rate=args.dr
normalize_weights=args.normalize_weights

# Define the training parameters
# num_epochs = 100
# learning_rate = 0.1 #0.001
# momentum = 0.5
# weight_decay = 0.001
input_size = 784
hidden_size= 256
# dropout_rate=0.5
weight_tying=True
logistic=False

if dataset_name == 'mnist':

    # # Define a series of transforms
    # transform = transforms.Compose([
    #     transforms.ToTensor(),              # Convert images to PyTorch tensors
    #     transforms.Normalize((0.1307,), (0.3081,))   # Typical std values for normalization
    # ])

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    input_size = 784
    num_classes = 10
elif dataset_name == 'cifar10':
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    input_size = 32 * 32 * 3
    num_classes = 10
elif dataset_name == 'random':
    rand_gen_mat = torch.linalg.svd(torch.Tensor(np.random.random((input_size,input_size))))[2][:hidden_size]
    train_dataset = TensorDataset(torch.randn((60000,hidden_size))@rand_gen_mat,torch.zeros(60000))
    test_dataset = TensorDataset(torch.randn((10000,hidden_size))@rand_gen_mat,torch.zeros(10000))

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(60,60))

def update_plot(features,nrow=16): # Alternate version
    with torch.no_grad():
        weight = features
        grid = torchvision.utils.make_grid(weight.unsqueeze(1), nrow=nrow, normalize=True, pad_value=1)
        plt.imshow(grid.permute(1, 2, 0))
    fig.suptitle('Feature Map')

# Define model
model = AutoEncoder(input_size=input_size,
                    dropout_rate=dropout_rate,
                    hidden_size=hidden_size,
                    weight_tying=weight_tying,
                    logistic=logistic,
                    normalize_weights=normalize_weights)

# Define the loss function and optimizer
if logistic:
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()
else:
    criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate/100, weight_decay=0.0)  # L2 Regularization

# Train the autoencoder
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1)

        # Forward pass
        decoded,encoded = model(data)

        # Compute the loss
        loss = criterion(decoded, data)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Max norm
        row_norms = max_norm(model.fc1,c=3.0)

        total_loss += loss.item()

    torch.set_printoptions(precision=2)
    print(row_norms.detach().numpy())
    torch.set_printoptions(precision=6)

    # Update and save the plot
    M = int(np.sqrt(hidden_size))
    image_side = int(np.sqrt(input_size))
    features = model.fc1.weight.view(-1,image_side,image_side).cpu()

    update_plot(features,nrow=M) 
    plt.savefig(f'feature_map.png')

    # Print the average loss for the epoch
    avg_loss = total_loss / (batch_idx + 1)
    mat = model.fc1.weight.t()@model.fc2.weight.t()
    frob = torch.linalg.matrix_norm(mat-torch.eye(input_size)).detach().numpy()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Frobenius: {frob:.6f}")

# Close the plot
plt.close()

