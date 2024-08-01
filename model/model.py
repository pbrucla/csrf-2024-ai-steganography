import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum

# from sklearn import NeuralNetBinaryClassifier

class ModelTypes(Enum):
    EfficientNet = 1

    SWIN = 3
    MobileNet = 4


def get_model(model_type: ModelTypes, num_classes: int) -> nn.Module:
    match model_type:
        case ModelTypes.EfficientNet:
            model = torchvision.models.efficientnet_v2_s(
                weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1"
            )
            model.classifier = nn.Sequential(
                nn.Dropout(0.2, inplace=True),
                nn.Linear(in_features=1280, out_features=num_classes),
            )

        case ModelTypes.SWIN:
            model = torchvision.models.swin_v2_t(
                weights="Swin_V2_T_Weights.DEFAULT"
            )
            model.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)

        case ModelTypes.MobileNet:
            model = torchvision.models.mobilenet_v3_small(
                weights='DEFAULT'
            )
            model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    return model


def freeze_model(model: nn.Module) -> None:
    for n, param in enumerate(model.parameters()):
        if n == 0 or n == 1:
            param.requires_grad = True
        else:
            param.requires_grad = False


# Unfreezes last layer of model with requires_grad = False
def unroll(model, optimizer, lr) -> None:
    for param in reversed(list(model.parameters())):
        if not param.requires_grad:
            param.requires_grad = True
            default_group = optimizer.param_groups[0].copy()
            default_group.update({"params": param, "lr": lr})
            optimizer.add_param_group(default_group)
            break


# Sets different learning rates for layers
def get_optimizer(model, base_lr, classifier_lr, unfrozen_layers=[0, 1, 2, 3]):
    classifier_params = []
    other_params = []
    for n, param in enumerate(model.parameters()):
        if param.requires_grad:
            if n in unfrozen_layers:
                classifier_params.append(param)
            else:
                other_params.append(param)

    parameter_lrs = []
    if classifier_params:
        parameter_lrs.append({"params": classifier_params, "lr": classifier_lr})
    if other_params:
        parameter_lrs.append({"params": other_params, "lr": base_lr})

    # Initialize optimizer?
    optimizer = torch.optim.AdamW(parameter_lrs)
    return optimizer


# def wrapper(model):
#     wrapped_model = NeuralNetBinaryClassifier(
#         model,
#         criterion=nn.CrossEntropyLoss,
#         optimizer=torch.optim.Adam,
#         # optimizer = torch.optim.AdamW(parameter_lrs)
#         lr=0.001,
#         max_epochs=9,
#         #batch_size=10
#         verbose=True
#     )
#     return wrapped_model