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
        "lr": [0.001, 0.0001, 0.00001],
        "max_epochs": [8, 10, 12, 14],
    }
    print("param_grid")

    # define model here
    model = wrapper(get_model(model_type, num_classes))
    print("model and wrapped")

    clf = GridSearchCV(model, param_grid, cv=5, scoring="accuracy", verbose=3, error_score='raise')
    print("clf")

    clf.fit(X, y)
    print("fit")

    params = clf.best_params_
    print("params")
    print(params)

    print("exited function")



if __name__ == "__main__":
    print("start")

    validation_dataset = Data(
        TrainingConfig.extract_lsb,
        enum_names_to_values(TrainingConfig.dataset_types),
        filepath=os.path.join("data", "val"),
        mode="val",
        down_sample_size=12,
    )

    print("class labels are " + str(validation_dataset.class_labels))

    def is_rgb(image):
        return image.shape[0] == 3

    # filtering out not-rgb
    filtered_dataset = [(image, label) for image, label in validation_dataset if is_rgb(image)]
    images, labels = zip(*filtered_dataset)

    X = torch.stack(images)
    y = torch.tensor(labels)

    get_efficient_hyperparameters(ModelTypes.EfficientNet, 2, X, y) 

    print("end")


