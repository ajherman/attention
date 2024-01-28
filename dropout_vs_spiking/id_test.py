# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# num_samples = 1000
# input_dimensions = 10
# num_epochs = 100
# lr_improved = 0.001
# input_size=10
# lr = 0.001
# def generate_white_noise_data(samples, dimensions):
#     return np.random.rand(samples, dimensions)
# data = generate_white_noise_data(num_samples, input_dimensions)



# class LinearAutoencoder(nn.Module):
#     def __init__(self, input_size):
#         super(LinearAutoencoder, self).__init__()
#         self.encoder = nn.Linear(input_size, input_size, bias=False)
#         self.decoder = nn.Linear(input_size, input_size, bias=False)
#         # Initializing weights to match NumPy implementation
#         nn.init.normal_(self.encoder.weight, mean=0.0, std=0.01)
#         nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# def train_autoencoder_pytorch(data, epochs, learning_rate):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LinearAutoencoder(input_size).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # Using SGD

#     dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size of 1

#     frobenius_norms_every_5th_epoch = []

#     for epoch in range(epochs):
#         for inputs, in dataloader:
#             inputs = inputs.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, inputs)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         if (epoch + 1) % 1 == 0:
#             with torch.no_grad():
#                 W = model.encoder.weight.data
#                 frobenius_norm = torch.norm(W @ W.T - torch.eye(input_size).to(device), p='fro').item()
#                 frobenius_norms_every_5th_epoch.append(frobenius_norm)

#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Frobenius Norm: {frobenius_norm:.4f}')

#     return frobenius_norms_every_5th_epoch

# # Training the PyTorch model with adjusted settings
# frobenius_norms_pytorch = train_autoencoder_pytorch(data, num_epochs, lr)


# print(frobenius_norms_pytorch)

import numpy as np

def generate_white_noise_data(samples, dimensions):
    return np.random.rand(samples, dimensions)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def train_autoencoder_frobenius_norm(data, epochs, learning_rate):
    # Normalize data
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Initialize weights
    input_size = data.shape[1]
    W = (np.random.randn(input_size, input_size)-0.5) * 0.01  # Smaller initial weights

    frobenius_norms = []

    for epoch in range(epochs):
        total_loss = 0

        for x in data:
            # Forward pass
            encoded = np.dot(x, W)
            decoded = np.dot(encoded, W.T)

            # Calculate loss
            loss = mse_loss(x, decoded)
            total_loss += loss

            # Backward pass - gradient descent
            grad = 2 * np.outer((decoded - x), encoded)
            W -= learning_rate * grad

        # Calculate and store Frobenius norm every 5th epoch
        if (epoch + 1) % 1 == 0:
            I = np.eye(input_size)
            frobenius_norm = np.linalg.norm(W @ W.T - I, 'fro')
            frobenius_norms.append(frobenius_norm)

    return frobenius_norms

# Parameters for the autoencoder and the white noise data
num_samples = 1000
input_dimensions = 200 #10
num_epochs = 20
lr_improved = 0.001

# Generate white noise data
data = generate_white_noise_data(num_samples, input_dimensions)

# Train the autoencoder and get Frobenius norms for every 5th epoch
frobenius_norms = train_autoencoder_frobenius_norm(data, num_epochs, lr_improved)

print(frobenius_norms)