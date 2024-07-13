import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

from model import get_model, get_optimizer, freeze_model, unroll
from test import test_one_epoch
from train import train_one_epoch
from stego_pvd import StegoPvd

import argparse
# import ModelTypes enum from model
from model import ModelTypes

from dataclasses import dataclass

@dataclass
class TrainingConfig:
    epochs: int = 2
    learning_rate: float = 0.001
    optimizer: str = "adamw"
    criterion: str = "bce_loss"
    model_type: ModelTypes = ModelTypes.EfficientNet
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN")
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-type', type=int, default=ModelTypes.EfficientNet, help='Model type: EfficientNet(1) or ResNet(2)')
    parser.add_argument('--criterion', type=str, default='bce_loss', help='Criterion type')
    parser.add_argument('--device', type=str, default='cpu', help='Device type: cpu, cuda, or mps')

    #Can you add an optimizer argument?
    return parser.parse_args()

def get_config():
    args = parse_args()
    return TrainingConfig(
        epochs = args.epochs,
        learning_rate = args.learning_rate,
        optimizer = args.optimizer,
        criterion = args.criterion,
        model_type = args.model_type,
        device = args.device
    )


if __name__ == "__main__":
    print("Starting Training")
    # create dataloaders here
    # https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    print("Creating datasets")

    #throw in an image
    train_dataset = StegoPvd(filepath=os.path.join( "Stego-pvd-dataset", "train"))
    test_dataset = StegoPvd(filepath=os.path.join("Stego-pvd-dataset", "test"), clean_path="cleanTest", stego_path="stegoTest")

    print("Creating DataLoaders")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    # it's throwing weird errors, not printing print (train_loader[0][0])
    #checking image formatted correctly
    for idx, (images, labels) in enumerate(train_loader):
        print(images[0])  # Get the first image from the batch
        check_image = images[0].transform()
        print ("\n transformed image: ", check_image)
        break 
    
    check_image = train_loader[1].transform()
    print (check_image)

    # visualize a sample from the train loader
    # train_iter = iter(train_loader)
    # batch_images, batch_labels = next(train_iter)

    # image, label = batch_images[0], batch_labels[0]
    # print(image.shape)
    # plt.imshow(image.permute(1,2,0))
    # plt.show()

    # Set up device for model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    LEARNING_RATE = 1e-3
    EPOCHS = 2

    print("Creating model")
    # create instance of model here
    model = get_model().to(device)
    freeze_model(model)
    optimizer = get_optimizer(model, LEARNING_RATE, LEARNING_RATE*2)
    criterion = nn.BCELoss()

    # train model for x epoches here (and run testing)model = 
    print("Starting model")
    for epoch in range(EPOCHS):
        train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
        # unroll(model)
        test_one_epoch(model, test_loader, device)