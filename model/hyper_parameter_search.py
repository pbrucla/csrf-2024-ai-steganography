# Make a function that runs a hyper parameter search by generating some configs that it uses to call train (in main)
# do grid search
# hyper parameters - learning rate, dataset type, maybe extract LSB
# store results in csv

import os
from sklearn.model_selection import GridSearchCV
from dataset import Data
from model import get_model, get_optimizer, freeze_model, unroll, ModelTypes, wrapper
import config
import torch
from config import enum_names_to_values
from main import TrainingConfig

def get_efficient_hyperparameters(model_type, num_classes, X, y):

    print("entered function")

    param_grid = {
        # later include more lr and more epochs (taking them out so faster testing)
        "lr": [0.001],
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

    clf.fit(X, y)
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
        down_sample_size=12,
    )

    def is_rgb(image):
        return image.shape[0] == 3

    filtered_dataset = [(image, label) for image, label in validation_dataset if is_rgb(image)]
    images, labels = zip(*filtered_dataset)

    X = torch.stack(images)
    y = torch.tensor(labels)

    get_efficient_hyperparameters(ModelTypes.EfficientNet, 7, X, y) 
    # Testing 1 as the first parameter to correspond to EfficientNet -- THIS IS THE PLAN

    #print(len(validation_dataset))
    #print(len(validation_dataset.labels))
    # print("validation_dataset[1]")
    #print(validation_dataset[0])
    #print("labels")
    #print(validation_dataset.labels[0])

    print("end")


