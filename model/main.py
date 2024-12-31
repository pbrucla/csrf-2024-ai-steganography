import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from dataset import get_datasets
from train import train_one_epoch
from test import test_one_epoch
from model import get_model, get_optimizer, freeze_model, unroll
from config import get_config

# for lr scheduler
from torch.optim.lr_scheduler import StepLR

def train_model(config, train_dataset, test_dataset, plot_data=False):
    print("Starting Training")

    print("Creating DataLoaders")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=True
    )

    print("Creating model")
    # create instance of model here
    model = get_model(config.model_type, len(config.dataset_types)).to(config.device)
    freeze_model(model)
    optimizer = get_optimizer(model, config.learning_rate, config.learning_rate * 2)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([
            sum(train_dataset.dataset_sizes) / i for i in train_dataset.dataset_sizes
        ]).to(config.device)
    )
    scheduler = StepLR(optimizer, config.step_size, config.gamma)
    class_labels = train_dataset.class_labels

    # train model for x epoches here (and run testing)model =
    print("Starting model")
    data_storage = []
    loss_values = []
    accu_values = []
    f1_scores = [] # list of lists (one f1 score per class)
    epoch_array = []

    for epoch in range(config.epochs):
        data_storage.append(train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, config.device, class_labels
        ))
        unroll(model, optimizer, config.learning_rate)
        test_statistics = test_one_epoch(model, test_loader, config.device, test_dataset.class_labels)
        scheduler.step()
        loss_values.append(data_storage[epoch][0])
        accu_values.append(data_storage[epoch][1])
        f1_scores.append(data_storage[epoch][2])
        epoch_array.append(epoch)

    if plot_data:
        # showing overall data trends for performance
        plt.subplot(2, 1, 1)
        plt.plot(epoch_array, loss_values, label="Loss over the Epochs", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.subplot(2, 1, 2)
        plt.plot(
            epoch_array, accu_values, label="Accuracy over the Epochs", color="green"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.show()

    return test_statistics, train_dataset.class_labels


if __name__ == "__main__":
    config = get_config()
    train_dataset, test_dataset = get_datasets(config)
    train_model(config, train_dataset, test_dataset)
