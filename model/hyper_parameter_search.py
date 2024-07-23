# Make a function that runs a hyper parameter search by generating some configs that it uses to call train (in main)
# do grid search
# hyper parameters - learning rate, dataset type, maybe extract LSB
# store results in csv

import os
from sklearn.model_selection import GridSearchCV
from dataset import Data
from model import get_model, get_optimizer, freeze_model, unroll
from enum import enum_names_to_values
from main import TrainingConfig

# def get_efficient_hyperparameters(X_train, Y_train):

param_grid = {
    'learning_rate': [0.001, 0.0001, 0.00001],
    'epochs': [9, 10, 11, 12],
    'extract_lsb': [] # implemet this after we figure everything else
}

# define model here
model = get_model(TrainingConfig.model_type, len(TrainingConfig.dataset_types)).to(TrainingConfig.device)
# wrap the model in Keras????? (using Keras)

clf = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")

# converted_dataset_types = enum_names_to_values(config.dataset_types)
val_dataset = Data(TrainingConfig.extract_lsb, enum_names_to_values(TrainingConfig.dataset_types), filepath=os.path.join("data", "val"), mode="val")

print(val_dataset.labels)

clf.fit(val_dataset)

params = clf.best_params_