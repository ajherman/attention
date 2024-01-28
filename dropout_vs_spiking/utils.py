import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class Classifier(nn.Module):
    def __init__(self, input_size=784, num_classes=10,dropout=0.5,layer_sizes=[784,8192,8192]):
        super(Classifier, self).__init__()
        self.layers=nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            self.layers.append(nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]),nn.ReLU(),nn.Dropout(dropout)))
        self.layers.append(nn.Linear(layer_sizes[-1], num_classes))
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
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

class AutoEncoder(nn.Module):
    def __init__(self,input_size=784,hidden_size=256,dropout_rate=0.5,weight_tying=True,logistic=True,normalize_weights=False):
        super(AutoEncoder, self).__init__()
        self.logistic = logistic
        if normalize_weights:
            print("Normalizing weights")
            self.unnormalized_fc1 = nn.Linear(input_size, hidden_size,bias=False)
            self.fc1 = weight_norm(self.unnormalized_fc1)
        else:
            print("Weights not normalized")
            self.fc1 = nn.Linear(input_size, hidden_size,bias=False) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size,bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

        if weight_tying:
            self.fc2.weight = nn.Parameter(self.fc1.weight.transpose(0, 1))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights using Gaussian distribution, biases to constant
        nn.init.normal_(self.fc1.weight, mean=0, std=1.0 / torch.sqrt(torch.tensor(self.fc1.in_features, dtype=torch.float)))
        # nn.init.normal_(self.fc1.weight, mean=0, std=0.05)

        # nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, mean=0, std=1.0 / torch.sqrt(torch.tensor(self.fc2.in_features, dtype=torch.float)))
        # nn.init.normal_(self.fc2.weight, mean=0, std=0.05)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.fc1(x)
        encoded = self.relu(encoded)
        dropout_encoded = self.dropout(encoded)
        decoded = self.fc2(dropout_encoded)
        if self.logistic:
            decoded = self.sigmoid(decoded)

        # x = self.encoder(x)
        # x = self.decoder(x)
        return decoded,encoded
    
# class AutoEncoder(nn.Module):
#     def __init__(self, input_size=784,hidden_size=256,dropout_rate=0.5):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(128, hidden_size),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.Linear(128, input_size),
#             nn.Sigmoid()  # Use Sigmoid to scale output to [0,1]
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

    
class ConvAutoEncoder(nn.Module):
    def __init__(self,input_size=(32,32,3),hidden_layer=512):
        super(ConvAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,bias=False)
        self.relu = nn.ReLU()
        self.conv_transpose = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1,bias=False)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv_transpose(x)
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
        

def max_norm(linear_layer,c=3.0):
    with torch.no_grad():
        # Compute the norm of each row (L2 norm by default)
        row_norms = linear_layer.weight.norm(p=2, dim=1, keepdim=True)

        # Avoid division by zero (add a small epsilon)
        # epsilon = 1e-6
        scale = torch.max(row_norms/c,torch.ones(row_norms.shape))

        # Normalize each row of the weight matrix
        linear_layer.weight.div_(scale)

    return row_norms.t()