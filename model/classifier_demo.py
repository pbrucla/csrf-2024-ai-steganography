import os
import torch
import matplotlib.pyplot as plt


def classifer_demo(dataloader, model):
    # select 10 random images from each label
    # clean, dct, fft, lsb, pvd, ssb4, ssbn
    # gather predictions from model
    # plot accuracy for each label

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx != 1:
            break


# Sample run
if __name__ == "__main__":
    print("Running demo")
    # run_demo()
    print("Demo finished")
