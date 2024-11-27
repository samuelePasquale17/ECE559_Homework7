# *****************************************************************************
# *****************************************************************************
# k-means using a neural network
# *****************************************************************************
# *****************************************************************************

# *****************************************************************************
# Preamble and dataset loading, based on PyTorch tutorial
# *****************************************************************************
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(device)
print(f"Using {device} device")

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64 # batch size equal to 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# *****************************************************************************
# Building the neural network
# *****************************************************************************
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Flatten layer to convert 28x28 images into 1D vectors of size 784
        self.flatten = nn.Flatten()
        # Define self.centers as a parameter of size 10x784 for cluster centers
        self.centers = nn.Parameter(torch.randn(10, 784))  # Random initialization
        # Softmax layer along dimension 1 (across clusters)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Step 1: Flatten the input and keep the result as x0 for later use
        x0 = self.flatten(x)

        # Step 2: Compute the surrogate for negative distance
        x = torch.matmul(x0, self.centers.t()) - 0.5 * torch.sum(self.centers ** 2, dim=1).flatten()

        # Step 3: Multiply by 20 and pass through the softmax layer
        x = 20 * x
        x = self.softmax(x)

        # Step 4: Reconstruct the input using the center (matrix multiplication with centers)
        x_reconstructed = torch.matmul(x, self.centers)

        # Step 5: Compute the error as the difference between reconstruction and original
        x_error = x_reconstructed - x0

        # Return the reconstruction error
        return x_error


model = NeuralNetwork().to(device)

# *****************************************************************************
# Train and test loops
# *****************************************************************************
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction (reconstruction error)
        pred = model(X.to(device))
        # Create a zero tensor of the same shape as the prediction
        zero_target = torch.zeros_like(pred)
        # Compute the loss by comparing the reconstruction error with zero
        loss = loss_fn(pred, zero_target)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluate the model
    with torch.no_grad():
        for X, y in dataloader:
            # Compute prediction (reconstruction error)
            pred = model(X.to(device))
            # Create a zero tensor of the same shape as the prediction
            zero_target = torch.zeros_like(pred)
            # Compute the loss by comparing the reconstruction error with zero
            test_loss += loss_fn(pred, zero_target).item()

    # Compute average loss
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f}")

# *****************************************************************************
# Optimization parameters and initialization
# *****************************************************************************
basic_train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
training_size = len(basic_train_dataloader.dataset)

# Loss function and optimizer
loss_fn = nn.MSELoss()  # Mean squared error
learning_rate = 4.5  # As specified in the question
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

"""
# Initialization (b)
with torch.no_grad():
    i = 0
    for X, y in basic_train_dataloader:
        if y == i:  # Select one image per digit
            model.centers[i, :] = X.flatten().to(device)  # Set the i-th center to the flattened image
            i += 1
        if i == 10:  # Stop after initializing all 10 centers
            break
"""

# Initialization (c)
# *****************************************************************************
# Initialization with uniform random sampling
# *****************************************************************************
with torch.no_grad():
    for i in range(10):
        # Generate a random center for each cluster
        model.centers[i, :] = torch.rand(28 * 28).to(device)
###



# *****************************************************************************
# Standard training epochs
# *****************************************************************************
print(model)
print("Training model...")
epochs = 10  # Train for 10 epochs as specified
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# *****************************************************************************
# Building the confusion matrix
# *****************************************************************************
print("Computing the confusion matrix...")
C = model.centers.detach().cpu()  # Extract the centers from the model
counts = torch.zeros((10, 10), dtype=torch.int32)  # Initialize a 10x10 matrix for counts


with torch.no_grad():
    for X, y in basic_train_dataloader:
        best_distance = 1e16  # Start with a very large distance
        best_index = 0  # Placeholder for the best cluster index
        for j in range(10):
            # Calculate the distance of X from the j-th center
            dist = torch.sum((X.flatten().to(device) - model.centers[j])**2).item()
            if dist < best_distance:  # Update if a closer center is found
                best_distance = dist
                best_index = j
        # Update the counts matrix at the (true label, predicted cluster) index
        counts[y.item(), best_index] += 1

print(counts.numpy())

# *****************************************************************************
# Displaying the centers
# *****************************************************************************
print("Cluster centers:")
for j in range(10):
    print(f"Cluster {j}")
    q = C[j].reshape(28, 28)  # Reshape the j-th center back to 28x28 for visualization
    plt.imshow(q, cmap="gray")  # Display the cluster center as an image
    plt.show()
