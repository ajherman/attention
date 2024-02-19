import torch
import torch.nn as nn
from utils import *
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Normalize
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import *

dataset = "MNIST"

if dataset == "MNIST":
    input_size = 784
    im_shape = (28, 28)
    # Download MNIST
    train_dataset = MNIST(root='~/datasets', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='~/datasets', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
elif dataset == "CIFAR10":
    input_size = 3072
    im_shape = (3, 32, 32)
    # Download CIFAR10
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.0,0.0,0.0), (1.0,1.0,1.0))
    # ])
    transform = ToTensor()
    train_dataset = CIFAR10(root='~/datasets', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='~/datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define static transformer
if dataset == "MNIST":
    # model = StaticMultiHeadAttention(dm=input_size, dk=112, dv=112, N=1000, heads=7, dropout=0.2, rectify=0)
    model = AltStaticMultiHeadAttention(dm=input_size, N=1000, heads=8, dropout=0.2, rectify=0)
elif dataset == "CIFAR10":
    # model = StaticMultiHeadAttention(dm=input_size, dk=3*128, dv=3*128, N=1000, heads=8, dropout=0.2, rectify=0)
    model = AltStaticMultiHeadAttention(dm=input_size, N=4000, heads=8, dropout=0.2, rectify=0)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train
for epoch in range(10):
    for itr, (x, _) in enumerate(train_loader):
        x = x.view(-1, input_size)
        # x = F.normalize(x, p=1, dim=1)  # Normalize the input
        y = model(x)
        # y = F.normalize(y, p=1, dim=1)  # Normalize the output
        loss = criterion(y, x)
        loss_scaled = loss.item() / torch.mean(x.pow(2)).item()  # Scale the loss by the average norm of x along the batch dimension
        
        if itr % 100 == 0:
            with torch.no_grad():
                index = torch.randint(0, y.size(0), (1,))
                y_sample = y[index].view(im_shape)
                x_sample = x[index].view(im_shape)
                sample = torch.cat((y_sample, x_sample), dim=1)

                # Save the image using matplotlib
                if dataset == "MNIST":
                    plt.imshow(sample.squeeze(), cmap='gray')
                elif dataset == "CIFAR10":
                    plt.imshow(sample.permute(1, 2, 0))
                plt.axis('off')
                plt.savefig('recon.png')
                plt.close() 
            
        # Print scaled loss and accuracy after each batch
        if itr % 50 == 0:
            print(f'Epoch {epoch}, Batch Loss (Scaled): {loss_scaled * 100:.2f}%')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  