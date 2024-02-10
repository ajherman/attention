import torch
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from transformers.utils import *

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

if dataset_name == 'mnist':
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    input_size = 784
    num_classes = 10
elif dataset_name == 'cifar10':
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    input_size = 32 * 32 * 3
    num_classes = 10

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

for dropout in [0.0,0.5]:
    # Create the classifier model
    model = Classifier(input_size,num_classes,dropout=dropout)
    # model = ConvNet(32,32,10)

    # Define the optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_error_list = []
    test_error_list = []
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            # Train eval
            train_correct, train_total = 0, 0
            for images, labels in train_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            # Test eval
            test_correct, test_total = 0, 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        test_accuracy = 100 * test_correct / test_total
        train_error = 100 - train_accuracy
        test_error = 100 - test_accuracy
        train_error_list.append(train_error)
        test_error_list.append(test_error)
        
        # Write to CSV file
        with open(csv_file, 'a') as f:
            f.write('{},{},{},{}\n'.format(epoch, train_accuracy, test_accuracy, dropout))

        print('Epoch: {}. Train Accuracy: {}. Test Accuracy: {}'.format(epoch, train_accuracy, test_accuracy))
        import matplotlib.pyplot as plt

    # Plot train and test accuracies
    plt.plot(train_error_list, label='Train Error; dropout='+str(dropout),color='red')
    plt.plot(test_error_list, label='Test Error; dropout='+str(dropout),color='blue')
    
plt.xlabel('Epoch')
plt.ylabel('Percent Error')
plt.title('Train and Test Errors')
plt.legend()
plt.savefig('train_test_errors.png')
plt.show()

print("Dataset:", dataset_name)
