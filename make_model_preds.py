#!/usr/bin/env python3

# %%
import torch
from fashionMNIST_NN import NeuralNetwork
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# %%
# Load the NN model FashionMNIST_Model.pth:
model = NeuralNetwork()
model.load_state_dict(torch.load('FashionMNIST_Model.pth'))

# %%
# Get the FashionMNIST test data:
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

# %%
# Prediction classes:
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[2][0], test_data[2][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


# %%
