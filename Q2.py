# *****************************************************************************
# *****************************************************************************
# Gaussian Autoencoder
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
        # Encoder: same as HW6's CNN, with added batch normalization and dropout
        self.Encoder = nn.Sequential(
            nn.Conv2d(1, 20, 4, 1),  # 28x28 -> 25x25
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            nn.Conv2d(20, 20, 4, 2),  # 25x25 -> 11x11
            nn.ReLU(),
            nn.MaxPool2d(2),  # 11x11 -> 5x5
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            nn.Flatten(),  # 20x5x5 -> 500
            nn.Linear(500, 250),  # 500 -> 250
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Dropout(0.1),
            nn.Linear(250, 10)  # 250 -> 10
        )

        # Decoder: inverted architecture with added layers
        self.Decoder = nn.Sequential(
            nn.Linear(10, 360),  # 10 -> 360
            nn.ReLU(),
            nn.BatchNorm1d(360),
            nn.Dropout(0.1),
            nn.Linear(360, 720),  # 360 -> 720
            nn.ReLU(),
            nn.BatchNorm1d(720),
            nn.Dropout(0.1),
            nn.Unflatten(1, (20, 6, 6)),  # 720 -> 20x6x6
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False),  # 20x6x6 -> 20x12x12
            nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=0, output_padding=1),  # 20x12x12 -> 20x28x28
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(20, 1, kernel_size=4, stride=1, padding=1),  # 20x28x28 -> 1x28x28
            nn.Sigmoid()  # Final output scaled between 0 and 1
        )

    def forward(self, x, enc_mode=1):
        # Step 1: Pass input through the encoder to get z (embedding)
        z = self.Encoder(x)  # Encoder generates the latent representation z

        # Step 2: Add Gaussian noise to z based on enc_mode
        z2 = enc_mode * z + (2 - enc_mode) * torch.randn(z.shape, device=z.device)

        # Step 3: Pass z2 through the decoder to get the reconstructed image f
        f = self.Decoder(z2)

        # Step 4: Calculate the reconstruction error
        e = f - x

        # Step 5: Flatten the reconstruction error (use nn.Flatten on the fly)
        e = e.view(e.size(0), -1)  # Flatten the reconstruction error

        # Step 6: Concatenate z (embedding) and e (reconstruction error)
        e = torch.cat([z, e], dim=1)

        return e
# *****************************************************************************
# Optimizing the neural network
# *****************************************************************************
# ...


model = NeuralNetwork().to(device)
# *****************************************************************************
# Train and test loops
# *****************************************************************************
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()
    for batch, (X, _) in enumerate(dataloader):  # We don't use labels
        # Compute prediction and loss
        pred = model(X.to(device), enc_mode=1)
        loss = loss_fn(pred, torch.zeros_like(pred))  # MSE Loss

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
        for X, _ in dataloader:
            pred = model(X.to(device), enc_mode=1)
            reconstruction_error = pred[:, 10:]  # Reconstruction error
            test_loss += loss_fn(reconstruction_error, torch.zeros_like(reconstruction_error)).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n {size} {num_batches}")

# *****************************************************************************
# Optimization parameters and initialization
# *****************************************************************************
loss_fn = nn.MSELoss()  # Mean squared error
learning_rate = 0.1  # Learning rate as specified
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
# Generating new images using the learned autoencoder
# *****************************************************************************
print("Generating new images...")
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for s in range(20):  # Generate 20 images
        # Generate random z values
        z_random = torch.randn(1, 10).to(device)  # Random embedding
        # Pass through the decoder only
        x = model.Decoder(z_random)
        imgX = x.reshape(28, 28).detach().cpu().numpy()  # Extract the image part
        plt.imshow(imgX, cmap="gray")
        plt.title(f"Generated Image {s+1}")
        plt.axis("off")
        plt.show()
