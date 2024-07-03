import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Write the test function here!

def test(model, test_loader, device):
    model.eval()