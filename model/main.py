import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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

# Set up device for model
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# create instance of model here
model = get_model()


# train model for x epoches here (and run testing)model = 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 2
for epoch in range(1, epochs):  
  train_one_epoch(model, train_loader, optimizer, criterion, device)
  test(model, test_loader, )
  pbar.update()

with tqdm(range(epochs)) as pbar:
    for epoch in pbar:
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        test(model, test_loader, )
        pbar.update()

ge, prediction, and label

