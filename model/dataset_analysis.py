import os
import numpy as np
from PIL import Image

def clean_pvd_diff():
    train_filepath = os.path.join("data", "train")

    clean_path = "cleanTrain"
    # dct_path = "DCTTrain"
    # fft_path = "FFTTrain"
    # lsb_path = "LSBTrain"
    pvd_path = "PVDTrain"
    # ssb4_path = "SSB4Train"
    # ssbn_path = "SSBNTrain"
    
    train_clean_filepaths = [os.path.join(train_filepath, clean_path, file) for file in os.listdir(path=os.path.join(train_filepath, clean_path))]
    # dct_filepaths = [os.path.join(train_filepath, dct_path, file) for file in os.listdir(path=os.path.join(train_filepath, dct_path))]
    # fft_filepaths = [os.path.join(train_filepath, fft_path, file) for file in os.listdir(path=os.path.join(train_filepath, fft_path))]
    # lsb_filepaths = [os.path.join(train_filepath, lsb_path, file) for file in os.listdir(path=os.path.join(train_filepath, lsb_path))]
    pvd_filepaths = [os.path.join(train_filepath, pvd_path, file) for file in os.listdir(path=os.path.join(train_filepath, pvd_path))]
    # ssb4_filepaths = [os.path.join(train_filepath, ssb4_path, file) for file in os.listdir(path=os.path.join(train_filepath, ssb4_path))]
    # ssbn_filepaths = [os.path.join(train_filepath, ssbn_path, file) for file in os.listdir(path=os.path.join(train_filepath, ssbn_path))]

    # train_stego_filepaths = dct_filepaths + fft_filepaths + lsb_filepaths + pvd_filepaths + ssb4_filepaths + ssbn_filepaths

    num = 1

    # skip the image named 0.jpg in cleanTrain
    train_clean_filepaths = sorted(train_clean_filepaths)[1:num+1]
    pvd_filepaths = sorted(pvd_filepaths)[:num]

    # np.set_printoptions(threshold=np.inf)

    pvd_diffs = []
    for clean, pvd in zip(train_clean_filepaths, pvd_filepaths):
        clean_image = Image.open(clean)
        pvd_image = Image.open(pvd)

        diff = np.abs(np.array(clean_image)[:, :, :-1] - np.array(pvd_image))
        # diff = np.abs(np.concatenate((np.array(pvd_image), np.zeros((512, 512, 1))), axis=2) - np.array(clean_image))
        pvd_diffs.append(diff)

        # pvd_diffs.append(np.max(np.array(clean_image)[:, :, :-1] - np.array(pvd_image)))

    return pvd_diffs

if __name__ == "__main__":
    print(len(clean_pvd_diff()))
    # clean_pvd_diff()
    # print(np.max(clean_pvd_diff()))  
