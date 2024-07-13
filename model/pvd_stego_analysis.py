import os
import numpy as np
from PIL import Image

def diff_stego_images():
    train_filepath=os.path.join( "Stego-pvd-dataset", "train")
    train_clean_filepaths = [os.path.join(train_filepath, "cleanTrain", file) for file in os.listdir(path=os.path.join(train_filepath, "cleanTrain"))]
    train_stego_filepaths = [os.path.join(train_filepath, "stegoTrain", file) for file in os.listdir(path=os.path.join(train_filepath, "stegoTrain"))]

    train_clean_filepaths = sorted(train_clean_filepaths)[:2000]
    train_stego_filepaths = sorted(train_stego_filepaths)[:2000]


    stego_diffs = []
    for clean, stego in zip(train_clean_filepaths, train_stego_filepaths):
        clean_image = Image.open(clean)
        stego_image = Image.open(stego)
        stego_diffs.append(np.max(np.array(clean_image)[:,:, :-1] - np.array(stego_image)))

    return stego_diffs

if __name__ == "__main__":
    print(np.max(diff_stego_images()))