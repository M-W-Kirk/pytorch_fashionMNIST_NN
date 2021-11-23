#!/usr/bin/env python3

# %%
# add modules:
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt 

# %%
# Download training data from the FashionMNIST dataset:
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# %%
# Download test data from the FashionMNIST dataset:
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# %%
# Set batch size of features and labels:
batch_size = 64

# Create data loaders:
train_dataloader = DataLoader(training_data, batch_size=batch_size)

test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# %%
# Create Pytorch NN model:

# Get CPU or GPU for training:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
# %%
# Define NN model:
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
# %%
# Set loss function & optimizer parameters:
# Use the cross-entropy distributions of the networks:
loss_fn = nn.CrossEntropyLoss()

# Use Stochastic Gradient Descent (SGD) to optimize the objective parameter:
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute the prediction error:
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropogate the prediction error to adjust the model parameters:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

