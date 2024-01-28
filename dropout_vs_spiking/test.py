import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as MNIST
import matplotlib.pyplot as plt
import numpy as np

# Define the model
class ThreeLayerFFN(nn.Module):
    def __init__(self, dropout_rate):
        super(ThreeLayerFFN, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x1 = torch.relu(self.layer1(x))
        x1_drop = self.dropout(x1)
        x2 = torch.relu(self.layer2(x1_drop))
        x2_drop = self.dropout(x2)
        x3 = self.layer3(x2_drop)
        return x3, x1, x2

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = MNIST.MNIST(root='~/datasets', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)

# Function to visualize histograms and feature maps
def visualize_activations(model, epoch, activation_data):
    for i, activation in enumerate(activation_data):
        act_vect = activation.numpy().ravel()
        sparsity = np.mean(act_vect<1e-10)
        plt.hist(act_vect, bins=100)
        plt.title(f"Layer {i+1} activations - Epoch {epoch}, Sparsity {100*sparsity} %")
        plt.show()

    # Visualize feature map of first layer
    with torch.no_grad():
        weight = model.layer1.weight.data.view(-1, 28, 28).cpu()
        grid = torchvision.utils.make_grid(weight.unsqueeze(1), nrow=16, normalize=True, pad_value=1)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title("Feature map - First Layer")
        plt.show()

# Training with different dropout rates
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
num_epochs = 5  # Change as needed

for dropout_rate in dropout_rates:
    print(f"Training with dropout rate: {dropout_rate}")
    model = ThreeLayerFFN(dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        activation_data = [[] for _ in range(2)]  # For two hidden layers

        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs, act1, act2 = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Collect activations
            activation_data[0].append(act1.detach().cpu())
            activation_data[1].append(act2.detach().cpu())

        # Concatenate all batch activations for each layer
        activation_data = [torch.cat(acts, 0) for acts in activation_data]
        visualize_activations(model, epoch, activation_data)
