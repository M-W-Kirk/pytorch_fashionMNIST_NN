#!/usr/bin/env python3
import torch
from fashionMNIST_NN import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load('FashionMNIST_Model.pth'))

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
x, y = test_data[0][0], test_data[0][1]
with torch.no_ggrad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

