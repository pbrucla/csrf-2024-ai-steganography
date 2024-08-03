import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import ModelTypes, get_model

# Using EfficientNet and 7 classes (clean, DCT, FFT, LSB, PVD, SSB4, SSBN)?
num_classes = 7
this_model = get_model(ModelTypes.EfficientNet, num_classes)

def fgsm_attack_func(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Experiment with a few different values
epsilons = [0, 0.1, 0.5]
ep_count = len(epsilons)


def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    # Adv example for later
    adv_examples = []

    for image, label in test_loader:
        image, label = image.to(device), label.to(device)
        image.requires_grad = True
        output = model(image)
        
        pass

# criterion = nn.CrossEntropyLoss(
#         weight=torch.tensor([
#             sum(train_dataset.dataset_sizes) / i for i in train_dataset.dataset_sizes
#         ]).to(config.device)
# )

