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
# import DatasetTypes enum from config 
from config import DatasetTypes
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    epochs: int = 2
    learning_rate: float = 0.001
    model_type: ModelTypes = ModelTypes.EfficientNet
    device: str = 'default' #if default"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    transfer_learning: bool = True
    extract_lsb: bool = False
    batch_size: int = 256
    dataset_types: tuple[str, ...] = ("CLEAN", "LSB")

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN")
    parser.add_argument('-e', '--epochs', type=int, default=2, help='Number of epochs to train')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model-type', type=int, default=ModelTypes.EfficientNet, help='Model type: EfficientNet(1) or ResNet(2)')
    parser.add_argument('--device', type=str, default='default', help='Device type: cpu, cuda, or mps')
    parser.add_argument('--transfer-learning', action='store_true', help='Enable model unrolling and freezing')
    parser.add_argument('--dataset-types', type=str, nargs='+', default=("CLEAN", "LSB"), choices=[i.name for i in DatasetTypes], help='Dataset type: CLEAN(1), DTC(2), FFT(4), LSB(8), PVD(16), SSB4(32), SSBN(64)')
    parser.add_argument('-el', '--extract-lsb', action='store_true', help='Enable masking bits for LSB')
    parser.add_argument('--batch-size', type=int, default=256)
    
    return parser.parse_args()

def get_device(device_argument):
    #if default set automatically
    if device_argument == 'default':
        if torch.cuda.is_available:
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    #if argument is not default return user inputted device
    return device_argument

def get_config():
    args = parse_args()

    return TrainingConfig(
        epochs = args.epochs,
        learning_rate = args.learning_rate,
        model_type = args.model_type,
        device = get_device(args.device), 
        transfer_learning = args.transfer_learning,
        extract_lsb = args.extract_lsb,
        batch_size = args.batch_size,
        dataset_types = args.dataset_types,
    )

#since the datset argument takes in a list of strings, this is used to convert that list back to integers for processing later
def enum_names_to_values(names):
    values = []
    for name in names:
        member = DatasetTypes[name]
        values.append(member.value)
    return values

def train_model(config):    
    print("Starting Training")

    # https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    print("Creating datasets")
    converted_dataset_types = enum_names_to_values(config.dataset_types)
    train_dataset = Data(config.extract_lsb, converted_dataset_types, filepath=os.path.join("data", "train"))
    test_dataset = Data(config.extract_lsb, converted_dataset_types, filepath=os.path.join("data", "test"), clean_path="cleanTest", lsb_path="LSBTest") #clean_path="cleanTest", dct_path="DCTTest", fft_path = "FFTTest", lsb_path = "LSBTest", pvd_path="PVDTest", ssb4_path = "SSB4Test", ssbn_path = "SSBNTest"

    print("Creating DataLoaders")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
 
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
    model = get_model(config.model_type).to(config.device)
    freeze_model(model)
    optimizer = get_optimizer(model, config.learning_rate, config.learning_rate*2)
    criterion = nn.BCELoss()

    # train model for x epoches here (and run testing)model = 
    print("Starting model")
    for epoch in range(config.epochs):
        train_one_epoch(epoch, model, train_loader, optimizer, criterion, config.device)
        unroll(model, optimizer, config.learning_rate)
        test_one_epoch(model, test_loader, config.device)


if __name__ == "__main__":
    config = get_config()
    train_model(config)