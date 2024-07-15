# Make a function that runs a hyper parameter search by generating some configs that it uses to call train (in main)
# do grid search
# hyper parameters - learning rate, dataset type, maybe extract LSB
# store results in csv

from sklearn.model_selection import GridSearchCV


def get_efficient_hyperparameters(X_train, Y_train):
    param_grid = {
        'learning_rate': [],
        'epochs': [],
        'extract_lsb': []
    }