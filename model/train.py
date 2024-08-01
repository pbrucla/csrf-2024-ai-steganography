import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataset import accuracy_metric


def train_one_epoch(
    epoch, model, train_loader, optimizer, criterion, device: str, class_labels
) -> None:
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
            predicted_classes = torch.argmax(batch_outputs, axis=1)
            new_correct, new_total = accuracy_metric(predicted_classes, batch_labels)
            correct += new_correct
            total += new_total

            # Calculate loss and do backpropagation
            loss = criterion(batch_outputs, batch_labels)
            loss.backward()

            # Do gradient descent
            optimizer.step()
            optimizer.zero_grad()

            
            status = {
                "loss": f"{loss.item():.4f}",
                "acc": f"{round(100 * correct / total, 3):.2f}%",
            }
            pbar.set_postfix(status)
            pbar.update()

    print(f"Training loss:  {round(loss.item(), 3)}")
    accuracy = round(100 * correct / total, 3)
    print(f"Training accuracy: {accuracy}%")
    data_baggage = [round(loss.item(), 3), accuracy, f1_scores]
    return data_baggage
