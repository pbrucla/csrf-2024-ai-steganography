import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from dataset import Data
from train import train_one_epoch
from test import test_one_epoch
from model import get_model, get_optimizer, freeze_model, unroll

# import ModelTypes enum from model
from model import ModelTypes

from dataclasses import dataclass
from config import DatasetTypes
from config import enum_names_to_values

# for lr scheduler
from torch.optim.lr_scheduler import StepLR


@dataclass
class TrainingConfig:
    epochs: int = 2
    learning_rate: float = 0.001
    model_type: ModelTypes = ModelTypes.EfficientNet
    device: str = "default"  # if default"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    transfer_learning: bool = True
    extract_lsb: bool = False
    batch_size: int = 256
    dataset_types: tuple[str, ...] = ("CLEAN", "LSB")
    step_size: int = 30
    gamma: float = 0.9


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN")
    parser.add_argument(
        "-e", "--epochs", type=int, default=2, help="Number of epochs to train"
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--model-type",
        type=int,
        default=ModelTypes.EfficientNet,
        help="Model type: EfficientNet(1) or ResNet(2)",
    )
    parser.add_argument(
        "--device", type=str, default="default", help="Device type: cpu, cuda, or mps"
    )
    parser.add_argument(
        "--transfer-learning",
        action="store_true",
        help="Enable model unrolling and freezing",
    )
    parser.add_argument(
        "--dataset-types",
        type=str,
        nargs="+",
        default=("CLEAN", "LSB"),
        choices=[i.name for i in DatasetTypes],
        help="Dataset type: CLEAN(1), DTC(2), FFT(4), LSB(8), PVD(16), SSB4(32), SSBN(64)",
    )
    parser.add_argument(
        "-el", "--extract-lsb", action="store_true", help="Enable masking bits for LSB"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--step-size", type=int, default=30, help="Specify step size for LR scheduler"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="Specify decay factor for LR scheduler"
    )

    return parser.parse_args()


def get_device(device_argument):
    # if default set automatically
    if device_argument == "default":
        if torch.cuda.is_available:
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    # if argument is not default return user inputted device
    return device_argument


def get_config():
    args = parse_args()
    # Dataset Checks
    assert len(args.dataset_types) > 1, "need more than one dataset"
    for dataset_type in args.dataset_types:
        assert dataset_type in [
            i.name for i in DatasetTypes
        ], f"{dataset_type} is not a valid dataset type"

    # User Input Checks
    assert args.epochs > 0, "# of epochs to train must be positive!"
    assert args.batch_size > 0, "Batch size must be a positive integer!"
    assert args.learning_rate > 0, "LR must be positive!"
    assert 0 < args.gamma < 1, "Gamma must be between 0 and 1!"
    assert args.device in ["cpu", "mps", "cuda", "default"] + [
        f"cuda:{n}" for n in range(8)
    ], "Specified device must be either cpu, cuda, or mps!"
    assert (
        args.model_type == ModelTypes.EfficientNet
        or args.model_type == ModelTypes.ResNet
    ), "Model type must either be EfficientNet(1) or ResNet(2)!"

    return TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        device=get_device(args.device),
        transfer_learning=args.transfer_learning,
        extract_lsb=args.extract_lsb,
        batch_size=args.batch_size,
        dataset_types=args.dataset_types,
        step_size=args.step_size,
        gamma=args.gamma,
    )


def train_model(config, plot_data=False):
    print("Starting Training")

    # https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    print("Creating datasets")
    converted_dataset_types = enum_names_to_values(config.dataset_types)
    train_dataset = Data(
        config.extract_lsb,
        converted_dataset_types,
        filepath=os.path.join("data", "train"),
        mode="train",
        down_sample_size=6000
    )
    test_dataset = Data(
        config.extract_lsb,
        converted_dataset_types,
        filepath=os.path.join("data", "test"),
        mode="test",
        down_sample_size=3000
    )

    print("Creating DataLoaders")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=True
    )

    # base_image = train_loader[1]
    # print ("prior image", base_image)
    # check_image = train_loader[1].transform()
    # print ("\n Changed image: ",check_image)

    # visualize a sample from the train loader
    # train_iter = iter(train_loader)
    # batch_images, batch_labels = next(train_iter)

    # image, label = batch_images[0], batch_labels[0]
    # print(image.shape)
    # plt.imshow(image.permute(1,2,0))
    # plt.show()
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

    # train model for x epoches here (and run testing)model =
    print("Starting model")
    data_storage = []
    loss_values = []
    accu_values = []
    epoch_array = []

    for epoch in range(config.epochs):
        data_storage.append(train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, config.device
        ))
        unroll(model, optimizer, config.learning_rate)
        test_one_epoch(model, test_loader, config.device)
        scheduler.step()
        loss_values.append(data_storage[epoch][0])
        accu_values.append(data_storage[epoch][1])
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

    return accu_values


if __name__ == "__main__":
    config = get_config()
    train_model(config)
