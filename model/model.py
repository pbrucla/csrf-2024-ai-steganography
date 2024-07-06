import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def get_model() -> nn.Module:
    model = torchvision.models.efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Dropout(.2, inplace=True),
        nn.Linear(in_features=1280, out_features=1),
        nn.Sigmoid()
    )
    return model

def freeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
    

# Unfreezes last layer of model with requires_grad = False
def unroll(model) -> None:
    for param in reversed(list(model.parameters())):
        if not param.requires_grad:
            param.requires_grad = True
            break

# Sets different learning rates for layers
def get_optimizer(model, base_lr, classifier_lr):

    parameters = [
        # {'params': model.features.parameters(), 'lr': base_lr},
        {'params': model.classifier.parameters(), 'lr': classifier_lr}
    ]

    # Initialize optimizer?
    optimizer = torch.optim.AdamW(parameters)
    return optimizer