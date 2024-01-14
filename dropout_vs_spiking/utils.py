import torch
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size=784, num_classes=10,dropout=0.5,layer_sizes=[784,8192,8192]):
        super(Classifier, self).__init__()
        self.layers=nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]),nn.ReLU(),nn.Dropout(dropout)))

        #     # self.add_module('fc'+str(i), nn.Sequential(nn.Linear(layer_sizes[i-1], layer_sizes[i]),nn.ReLU(),nn.Dropout(dropout)))
        # self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_layer),nn.ReLU(),nn.Dropout(dropout))
        # # self.fc2 = nn.Linear(512, 512)
        # self.fc2 = nn.Sequential(nn.Linear(512, 512),nn.ReLU(),nn.Dropout(dropout))
        # # self.fc3 = nn.Linear(512, 512)
        # self.fc3 = nn.Sequential(nn.Linear(512, 512),nn.ReLU(),nn.Dropout(dropout))
        # # self.fc4 = nn.Linear(512, num_classes)
        self.layers.append(nn.Linear(layer_sizes[-1], num_classes))
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        # x = self.(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        # x = self.dropout(torch.relu(self.fc1(x)))
        # x = self.dropout(torch.relu(self.fc2(x)))
        # x = self.dropout(torch.relu(self.fc3(x)))
        # x = self.fc4(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, input_size_x,input_size_y, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the linear layer input based on the input image size
        self.fc_input_size = 32 * (input_size_x // 4) * (input_size_y // 4)
        
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomDropout(nn.Module):
    def __init__(self, p=0.5):
        super(CustomDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full(x.size(), 1 - self.p)).to(x.device)
            return x * mask / (1 - self.p)
        else:
            return x