import os
import torch
import matplotlib.pyplot as plt
from display_images import load_random_images
import re
from main import test_dataset
# delete when done (all imports below)
from dataset import Data
import config
from enum_utils import enum_names_to_values


def classifer_demo(directories, dataset,model, num_images = 10):
    # select 10 random images from each label
    # gather predictions from model
    # plot accuracy for each label

    print("running")
    print(dataset.class_labels)
    # just so i can see what they are

    labels = []
    accuracies = []

    for directory in directories:

        true_label = "os.path.__"
        labels.append(true_label)
        # collect labels from directory name and append to labels

        images = load_random_images(directory, num_images)

        predictions = model(images).squeeze()
        
        num_correct = 0
        for item in predictions:
        # caclulate accuracy
            predicted_label = labels[torch.argmax(item)]
            if predicted_label==true_label:
                num_correct += 1

        #calculate and store the accuracy for each steg type
        accuracy = num_correct/num_images if num_images > 0 else 0 
        accuracies.append(accuracy)

    # plot accuracy for each label on the same graph
    plt.title("Classifier Demo Accuracy for Different Labels")
    plt.bar(labels, accuracies)
    plt.xlabel("Steg Type")
    plt.ylabel("Accuracy")

    #display text for all the labels
    for index, value in enumerate (accuracies):
        plt.text(index, value, f"{round(value*100, 2)}%", ha = "center", va = "bottom")

    plt.show()

    

        

#Sample run
if __name__ == "__main__":
    print("Running demo")

    directories = ["hi"]

    # delete when done start
    converted_dataset_types = enum_names_to_values(config.dataset_types)

    test_dataset = Data(config.extract_lsb, converted_dataset_types, filepath=os.path.join("data", "test"), mode="test")
    # delete when done end

    classifer_demo(directories, test_dataset, "model")

    print("Demo finished")

