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

    num = 10

    # skip the image named 0.jpg in cleanTrain
    train_clean_filepaths = sorted(train_clean_filepaths)[1:num+1]
    pvd_filepaths = sorted(pvd_filepaths)[:num]

    pvd_diffs = []
    avg_pvd_diffs = []
    max_pvd_diffs = []

    for clean, pvd in zip(train_clean_filepaths, pvd_filepaths):
        clean_image = Image.open(clean)
        pvd_image = Image.open(pvd)

        pvd_diff = np.abs(np.array(clean_image)[:, :, :-1] - np.array(pvd_image))
        # pvd_diff = pvd_diff[pvd_diff != 255]
        pvd_diffs.append(pvd_diff)

        avg_pvd_diff = np.mean(pvd_diff)
        avg_pvd_diffs.append(avg_pvd_diff)

        max_pvd_diff = np.max(pvd_diff)
        max_pvd_diffs.append(max_pvd_diff)

    # rgb = ["Red", "Green", "Blue"]

    # for i, row in enumerate(pvd_diff):
    #     for j, col in enumerate(row):
    #         for k, channel in enumerate(col):
    #             if channel == 0 or channel == 255:
    #                 continue
    #             print(f"Row: {i}, Column: {j}, {rgb[k]}: {channel}")

    return pvd_diffs, avg_pvd_diffs, max_pvd_diffs

def main():
    all_diffs = clean_pvd_diff()
    for avg, max in zip(all_diffs[1], all_diffs[2]):
        print(f"Average pixel difference: {avg}")
        print(f"Max pixel difference: {max}")

if __name__ == "__main__":
    main()