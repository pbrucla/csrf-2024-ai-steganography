# Make a function that runs a hyper parameter search by generating some configs that it uses to call train (in main)
# do grid search
# hyper parameters - learning rate, dataset type, maybe extract LSB
# store results in csv

import os
from sklearn.model_selection import GridSearchCV
from dataset import Data
from model import get_model, get_optimizer, freeze_model, unroll, ModelTypes, wrapper
import config
from config import enum_names_to_values
from main import TrainingConfig


def get_efficient_hyperparameters(model_type, num_classes, val_dataset):

    print("entered function")

    param_grid = {
        "lr": [0.001, 0.0001, 0.00001],
        "max_epochs": [10, 12],
    }
    # extract_lsb": []  # implemet this after we figure everything else
    print("param_grid")

    # define model here
    model = wrapper(get_model(model_type, num_classes))
    print("model and wrapped")

    clf = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", verbose=3)
    print("clf")

    #print("val_dataset.labels:")
    #print(val_dataset.labels)  

    clf.fit(val_dataset, val_dataset.labels)
    print("fit")

    params = clf.best_params_
    print("params")
    print(params)

    print("exited function")



# Sample run
if __name__ == "__main__":
    print("start")

    validation_dataset = Data(
        TrainingConfig.extract_lsb,
        enum_names_to_values(TrainingConfig.dataset_types),
        filepath=os.path.join("data", "val"),
        mode="val",
    )

    get_efficient_hyperparameters(ModelTypes.EfficientNet, 7, validation_dataset) 
    # Testing 1 as the first parameter to correspond to EfficientNet -- THIS IS THE PLAN

    print("end")


