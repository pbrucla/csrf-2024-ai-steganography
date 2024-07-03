import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Write the train function here!

def train(model, train_loader, optimizer, criterion, device):
    model.train()