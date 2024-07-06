import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Write the train function here!
def train_one_epoch(epoch, model, train_loader, optimizer, criterion, device: str) -> None:
    
    # Set model to training model
    model.train()
    
    correct = 0
    total = 0
    with tqdm(range(len(train_loader))) as pbar:
        for batch_images, batch_labels in train_loader:
            pbar.set_description(f"Epoch {epoch+1}")

            # Send data to device
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # Run forward pass
            batch_outputs = model(batch_images).squeeze()

            # Calculate accuracy statistics batch_out
            correct += (batch_outputs.round() == batch_labels).sum(dtype=torch.int).item()
            total += len(batch_outputs)

            # Calculate loss and do backpropagation
            loss = criterion(batch_outputs, batch_labels)
            loss.backward()
            
            # Do gradient descent
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{round(100 * correct / total, 3):.2f}%"})
            pbar.update()

    print(f'Training loss:  {round(loss.item(), 3)}')
    accuracy = round(100 * correct / total, 3)
    print(f'Training accuracy: {accuracy}%')
