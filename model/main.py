import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

from model import get_model, get_optimizer, freeze_model, unroll
from test import test_one_epoch
from train import train_one_epoch

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

# Set up device for model
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

LEARNING_RATE = 1e-4
EPOCHS = 2

# create instance of model here
model = get_model()
freeze_model(model)
optimizer = get_optimizer(model, LEARNING_RATE, LEARNING_RATE/2)
criterion = nn.BCELoss()

# train model for x epoches here (and run testing)model = 
with tqdm(range(EPOCHS)) as pbar:
    for epoch in pbar:
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_one_epoch(model, test_loader, device)
        pbar.update()