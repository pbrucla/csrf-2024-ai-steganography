import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum

class ModelTypes(Enum):
    EfficentNet = 1
    ResNet = 2

def get_model(model_type: ModelTypes) -> nn.Module:
    match model_type:
        case ModelTypes.EfficentNet:
            model = torchvision.models.efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1")
            model.classifier = nn.Sequential(
                nn.Dropout(.2, inplace=True),
                nn.Linear(in_features=1280, out_features=1),
                nn.Sigmoid()
            )

        case ModelTypes.ResNet:
            model = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
            model.fc = nn.Sequential(
                nn.Linear(in_features=512, out_features=1, bias=True),
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