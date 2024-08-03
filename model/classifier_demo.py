import os
import torch
import matplotlib.pyplot as plt
from display_images import load_random_images
import re
# delete when done (all imports below)
from dataset import Data
import config
from config import enum_names_to_values, DatasetTypes
from model import ModelTypes, get_model

def classifer_demo(directories, dataset, model, num_images = 10):
    # select 10 random images from each label
    # gather predictions from model
    # plot accuracy for each label
    # labels are ['clean', 'DCT', 'FFT', 'LSB', 'PVD', 'SSB4', 'SSBN']

    # print("running")

    labels = []
    accuracies = []

    # print("hit directories loop")

    for directory in directories:

        # collect labels from directory name and append to labels
        true_label = path_to_label(directory)[0]
        # print("True label is " + true_label)
        labels.append(true_label)

        print("labelling done")

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

def path_to_label(path):
    pathstr = str(path)
    pattern = r"data/(test|train|val)/|Test/|Train/|Val/"
    subsitution = ""

    label = re.sub(pattern, subsitution, pathstr)

    if label == "clean":
        return "Clean",

    return label
        

# Sample run
if __name__ == "__main__":
    print("Running demo")

    directories = ["data/test/cleanTest/", "data/test/DCTTest/"]

    # delete when done start
    dataset_types = [DatasetTypes.CLEAN, DatasetTypes.DCT, DatasetTypes.FFT, DatasetTypes.LSB, DatasetTypes.PVD, DatasetTypes.SSB4, DatasetTypes.SSBN]

    test_dataset = Data(
        False, 
        dataset_types, 
        filepath=os.path.join("data", "test"), 
        mode="test"
    )
    # delete when done end

    # print("test_dataset.labels is " + str(test_dataset.labels))

    model = get_model(ModelTypes.EfficientNet, 2)

    classifer_demo(directories, test_dataset, model)

    print("Demo finished")
