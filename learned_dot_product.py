import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # layers.append(nn.Sigmoid())
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout_rate))
        # layers.append(nn.Linear(1,1,bias=False))
        # layers.append(torch.squeeze())
        
        self.model = nn.Sequential(*layers)

    def forward(self,x,y):
        z = torch.cat((x,y),1)
        out= self.model(z)
        return torch.squeeze(out)

# Define the dot product loss function
def dot_product_loss(output, target):
    # return nn.L1Loss()(output, target)
    return nn.MSELoss()(output, target)


# Set the random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define the training parameters
dk = 64
N = 1000
layer_sizes = [2 * dk,20, 1]
dropout_rate = 0.5
learning_rate = 1e-4
num_iterations = 2000

# Generate random vectors for training
# vectors = [torch.randn(dk) for _ in range(N)]

# Create the MLP model
model = MLP(layer_sizes, dropout_rate)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = dot_product_loss

# Train the model
errors = []
for iteration in range(num_iterations):

    # batch_vectors = random.choices(vectors, k=N)
    x_vectors = [torch.abs(torch.randn(dk)/torch.sqrt(torch.tensor(dk))) for _ in range(N)]
    y_vectors = [torch.abs(torch.randn(dk)/torch.sqrt(torch.tensor(dk))) for _ in range(N)]
    targets = [torch.mean(x_vectors[i]*y_vectors[i]) for i in range(N)]
    # Print lengths of x and y vectors and targets
    
    # END: ed8c6549bwf9
    x_input_batch = torch.stack(x_vectors)
    y_input_batch = torch.stack(y_vectors)
    target_batch = torch.stack(targets)
    
    if iteration == 0:
        x_norms = torch.stack([torch.norm(x) for x in x_vectors])
        # target_norms = [torch.norm(t) for t in targets]
        
        plt.hist(x_norms.tolist(), bins=20)
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')
        plt.title('Histogram of X Vector L2 Norms')
        plt.show()
        # print(target_batch)
        plt.hist(target_batch.tolist(), bins=20)
        plt.xlabel('L2 Norm')
        plt.ylabel('Frequency')
        plt.title('Histogram of Target L2 Norms')
        plt.show()

    optimizer.zero_grad()
    output = model(x_input_batch,y_input_batch)
    # print(output.size())
    # print(target_batch.size())
    # assert(0)
    loss = loss_fn(output, target_batch)
    loss.backward()
    optimizer.step()

    errors.append(loss.item())
# Generate the error graph
plt.plot(range(num_iterations), errors)
plt.xlabel('Training Iterations')
plt.ylabel('Error')
min_error = min(errors)
plt.title(f'Error vs. Training Iterations (Min Error: {min_error:.4f})')
plt.show()


