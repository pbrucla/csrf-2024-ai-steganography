import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Write the test function here!

def test_one_epoch(model, test_loader, device : str):
    #counters for both correct and total predictions
    correct = 0
    sum = 0
    with torch.no_grad(): #does not calculate gradients for performance optimization (does not store gradient graphs)
        model.eval() #puts into an evaluation state (drops droput layer and changes normalization layer)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(axis=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    #calculate and print out accuracy
    accuracy = round(100 * correct / total, 3)
    print(f'Accuracy at end of epoch: {accuracy}%')
    