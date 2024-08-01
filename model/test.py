import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataset import accuracy_metric

# Write the test function here!

<<<<<<< HEAD

def test_one_epoch(model, test_loader, device: str, class_labels):
    # counters for both correct and total predictions
=======
def test_one_epoch(model, test_loader, device : str, class_labels):
    #counters for both correct and total predictions
>>>>>>> f94a04aab762f121c1b8546b67594660ab934f14
    correct = 0
    total = 0
    all_f1_scores = []
    with torch.no_grad():  # does not calculate gradients for performance optimization (does not store gradient graphs)
        model.eval()  # puts into an evaluation state (drops droput layer and changes normalization layer)

        with tqdm(range(len(test_loader))) as pbar:
            for batch_inputs, batch_labels in test_loader:
                batch_inputs, batch_labels = (
                    batch_inputs.to(device),
                    batch_labels.to(device),
                )
                batch_outputs = F.sigmoid(model(batch_inputs)).squeeze()

                predicted_classes = torch.argmax(batch_outputs, axis=1)
                new_correct, new_total = accuracy_metric(predicted_classes, batch_labels)
                correct += new_correct
                total += new_total
<<<<<<< HEAD
                f1_scores = f1_score(batch_labels, predicted_classes, average=None)
=======
                f1_scores = f1_score(batch_labels.cpu(), predicted_classes.cpu(), average=None)

                all_f1_scores.append(f1_scores)
>>>>>>> f94a04aab762f121c1b8546b67594660ab934f14

                status = {"acc": f"{round(100 * correct / total, 3):.2f}%"}
                for class_label, f1 in zip(class_labels, f1_scores):
                    # idk if test_loader. is correct
                    status[class_label] = f1
                pbar.set_postfix(status)
                pbar.update()

    # calculate and print out accuracy
    accuracy = round(100 * correct / total, 3)
    print(f"Accuracy at end of epoch: {accuracy}%")

    average_f1 = sum(all_f1_scores) / len(all_f1_scores) 

    return accuracy, average_f1