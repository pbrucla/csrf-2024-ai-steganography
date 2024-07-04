import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Write the train function here!

def train_one_epoch(model, train_loader, optimizer, criterion, device: str) -> None:
    model.train()
    
    for image, label in train_loader:
        # Send data to device
        image = image.to(device)
        label = label.to(device)

        # Run forward pass
        output = model(image)

        # Calculate loss and do backpropagation
        loss = criterion(output, label)
        loss.backward()

        # Do gradient descent
        optimizer.step()
        optimizer.zero_grad()
    print('End of epoch loss:', round(loss.item(), 3))
                
