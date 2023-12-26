import torch
from torchvision import datasets, transforms

# Download MNIST dataset and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Convert dataset into tensors
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



class softmaxAutoEncoder:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.output_size = output_size
        self.net = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                                       torch.nn.Softmax(dim=1),
                                       torch.nn.Linear(hidden_size, input_size))
    def forward(self, x):
        return self.net(x)

def train(self, train_loader, test_loader, epochs=10, lr=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, self.input_size)
            optimizer.zero_grad()
            outputs = self.forward(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    return self.net 

def test(self, test_loader):
    criterion = torch.nn.MSELoss()
    test_loss = 0
    with torch.no_grad():
        for batch_features, _ in test_loader:
            batch_features = batch_features.view(-1, self.input_size)
            outputs = self.forward(batch_features)
            test_loss += criterion(outputs, batch_features).item()
    test_loss /= len(test_loader)
    print(f'Test loss: {test_loss:.4f}')
    return test_loss

def main():
    input_size = 784
    hidden_size = 32
    # output_size = 784
    model = softmaxAutoEncoder(input_size, hidden_size)
    model = train(model, train_loader, test_loader)
    test(model, test_loader)