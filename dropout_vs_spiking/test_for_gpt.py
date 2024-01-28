import torch
import argparse
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
# from utils import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
W_init = 0.01*(np.random.random((20,20))-0.5)

# Define the neural network architecture

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
parser.add_argument('--data-dir', type=str, default='~/datasets', help='Directory to store data')
parser.add_argument('--csv-file', type=str, default='data.csv', help='CSV file location')

args = parser.parse_args()

dataset_name = args.dataset
device = args.device
data_dir = args.data_dir
csv_file = args.csv_file


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
input_size = 784
num_classes = 10


# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the training parameters
num_epochs = 1000
learning_rate = 0.001 #0.001
momentum = 0.5
initial_momentum = 0.5
weight_decay = 0.001
input_size = 20
hidden_size=20 

class AutoEncoder(nn.Module):
    def __init__(self,input_size=20,hidden_size=20,dropout_rate=0.0,weight_tying=True):
        super(AutoEncoder, self).__init__()
        
        # # Version 1
        self.fc1 = nn.Linear(input_size, hidden_size,bias=False) #weight_norm(self.unnormalized_fc1)
        # self.fc1.weight = nn.Parameter(torch.Tensor(W_init.transpose()))
        self.fc2 = nn.Linear(hidden_size, input_size,bias=False)
        # self.fc2.weight = nn.Parameter(self.fc1.weight.t())

        # # Version 2
        # self.W1 = nn.Parameter(torch.Tensor(W_init))
        # self.W2 = nn.Parameter(self.W1.t())



    def forward(self, x):
        # Version 1
        encoded = self.fc1(x)
        decoded = self.fc2(encoded)
        # encoded = x@self.fc1.weight
        # decoded = encoded@self.fc2.weight

        # # Version 2
        # encoded = x@self.W1
        # decoded = encoded@self.W2

        return decoded,encoded
    

# Define model
model = AutoEncoder(input_size=input_size,dropout_rate=0.0,hidden_size=hidden_size,weight_tying=True)

# Train the autoencoder
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (_, _) in enumerate(train_loader):
        # data = data.view(data.size(0), -1)
        
        data = (torch.rand((1,20))-0.5)/0.288675134

        data = data.to(device)

        # Temporary test 2 
        with torch.no_grad():
            decoded,encoded = model(data)
            grad = 2*(decoded-data).t()@encoded # batch_size 64 #torch.outer(output-data,encoded)
            learning_rate = 0.001

            # Version 1
            model.fc1.weight = nn.Parameter(model.fc1.weight-learning_rate*grad.t())
            model.fc2.weight = nn.Parameter(model.fc1.weight.t())

            # # Version 2
            # model.W1-=learning_rate*grad
            # model.W2=nn.Parameter(model.W1.t())


    # # Version 1
    mat = model.fc1.weight.t()@model.fc2.weight.t()

    # Version 2
    # mat = model.W1@model.W2

    frob = torch.linalg.matrix_norm(mat-torch.eye(20))
    # frob = torch.linalg.matrix_norm(W@W.t()-torch.eye(20))
    print("MSE: ",frob.detach().numpy())











# import torch
# import argparse
# import torchvision
# from torchvision import datasets, transforms
# import torch.nn as nn
# import torch.optim as optim
# # from utils import *
# import numpy as np
# import matplotlib.pyplot as plt
# np.random.seed(1234)
# W_init=(np.random.random((20,20))-0.5)*0.01


# # Define the neural network architecture

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
# parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
# parser.add_argument('--data-dir', type=str, default='~/datasets', help='Directory to store data')
# parser.add_argument('--csv-file', type=str, default='data.csv', help='CSV file location')

# args = parser.parse_args()

# dataset_name = args.dataset
# device = args.device
# data_dir = args.data_dir
# csv_file = args.csv_file



# transform = transforms.ToTensor()

# train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
# input_size = 784
# num_classes = 10

# # Create data loaders
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# # Define the training parameters
# num_epochs = 1000
# learning_rate = 0.001 #0.001
# momentum = 0.5
# initial_momentum = 0.5
# weight_decay = 0.001
# input_size = 20
# hidden_size=20 

# class AutoEncoder(nn.Module):
#     def __init__(self,input_size=20,hidden_size=20,dropout_rate=0.0,weight_tying=True):
#         super(AutoEncoder, self).__init__()
#         # self.W = nn.Parameter((torch.Tensor(W_init)))
#         self.W = torch.Tensor(W_init)

#     def forward(self, x):
#         encoded = x@self.W
#         decoded = encoded@self.W.t() 
#         return decoded,encoded
    

# # Define model
# model = AutoEncoder(input_size=input_size,dropout_rate=0.0,hidden_size=hidden_size,weight_tying=True)
# W = torch.Tensor(W_init) 

# # Define the loss function and optimizer

# # Train the autoencoder
# for epoch in range(num_epochs):
#     total_loss = 0
#     for batch_idx, (_, _) in enumerate(train_loader):
        
#         data = (torch.rand((1,20))-0.5)/0.288675134

#         data = data.to(device)

#         # Version 2
#         with torch.no_grad():
#             decoded,encoded = model(data)
#             grad = 2*(decoded-data).t()@encoded # batch_size 64 #torch.outer(output-data,encoded)
#             learning_rate = 0.001
#             model.W-=learning_rate*grad

#         # # Version 1
#         # encoded = data@W
#         # decoded = encoded@W.t()
#         # grad = 2*(decoded-data).t()@encoded
#         # learning_rate=0.001
#         # W-=learning_rate*grad
    

#     # mat = W@W.t() # Version 1
#     mat = model.W@model.W.t() # Version 2
#     frob = torch.linalg.matrix_norm(mat-torch.eye(20))
#     # frob = torch.linalg.matrix_norm(W@W.t()-torch.eye(20))
#     print("MSE: ",frob.detach().numpy())






# (torch.rand((20,20),device=device)-0.5)*0.01





# # import torch
# # import argparse
# # import torchvision
# # from torchvision import datasets, transforms
# # import torch.nn as nn
# # import torch.optim as optim
# # # from utils import *
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Define the neural network architecture

# # parser = argparse.ArgumentParser()
# # parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
# # parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
# # parser.add_argument('--data-dir', type=str, default='~/datasets', help='Directory to store data')
# # parser.add_argument('--csv-file', type=str, default='data.csv', help='CSV file location')

# # args = parser.parse_args()

# # dataset_name = args.dataset
# # device = args.device
# # data_dir = args.data_dir
# # csv_file = args.csv_file



# # transform = transforms.ToTensor()
# # train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
# # test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
# # input_size = 784
# # num_classes = 10


# # # Create data loaders
# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# # # Create a figure and axis for the plot
# # fig, ax = plt.subplots(figsize=(60,60))

# # # Define the training parameters
# # num_epochs = 1000
# # learning_rate = 0.001 #0.001
# # momentum = 0.5
# # initial_momentum = 0.5
# # weight_decay = 0.001
# # input_size = 20
# # hidden_size=20 

# # W = (torch.rand((20,20),device=device)-0.5)*0.01

# # # Train the autoencoder
# # for epoch in range(num_epochs):
# #     total_loss = 0

# #     # Try switching which of these lines is commeneted
# #     for batch_idx, (data, _) in enumerate(train_loader):
# #     # for batch_idx in range(938):
        
# #         data = (torch.rand((1,20))-0.5)/0.288675134

# #         data = data.to(device)

# #         # Temporary test
# #         encoded = data@W
# #         decoded = encoded@W.t()
# #         grad = 2*(decoded-data).t()@encoded
# #         learning_rate=0.001
# #         W-=learning_rate*grad
# #     frob = torch.linalg.matrix_norm(W@W.t()-torch.eye(20))
# #     print("MSE: ",frob.detach().numpy())



