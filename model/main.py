import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os

import model
import test
import train

# create dataloaders here
# https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
train_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join("..", "Stego-pvd-dataset", "train"),
    transform= transforms.ToTensor()
)

test_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join("..", "Stego-pvd-dataset", "test"),
    transform= transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

# visualize a sample from the train loader
train_iter = iter(train_loader)
batch_images, batch_labels = next(train_iter)

image, label = batch_images[0], batch_labels[0]
print(image.shape)
plt.imshow(image.permute(1,2,0))
plt.show()

# create instance of model here

# train model for x epoches here (and run testing)

# run a random image through the model, output image, prediction, and label

