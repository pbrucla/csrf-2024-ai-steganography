from PIL import Image
from collections import defaultdict
import os
import numpy as np
from pathlib import Path

def get_image_info(image_path, filetype_dictionary, mode_dictionary, size_dictionary):
    if Path(image_path).suffix.lower() == ".txt":
        print(f"\nBad filepath: {image_path}\n")
        return
    with Image.open(image_path) as img:
        filetype = img.format
        filetype_dictionary[filetype] += 1
        mode = img.mode
        mode_dictionary[mode] += 1
        width, height = img.size
        size_dictionary[(width, height)] += 1

def print_dictionary(dict: dict[str, int], root_str : str) -> None:
    print(root_str, end=' ')
    for key, val in dict.items():
        print(key, ": ", val, end='   ')
    print('\n')

def print_rgba_values(image_path):
    with Image.open(image_path) as img:
        if (img.mode == 'RGBA'):
            red = img.getchannel('R')
            green = img.getchannel('G')
            blue = img.getchannel('B')
            alpha = img.getchannel('A')
            for y in range(alpha.height):
                for x in range(alpha.width):
                    print(f"({red.getpixel((x, y))}, {green.getpixel((x, y))}, {blue.getpixel((x, y))}, {alpha.getpixel((x, y))})", end=' ')
                    print()

def get_file_info(datasets: dict, get_filetypes=True, get_modes=True, get_dimensions=True):
    for name, filepaths in datasets.items():
        filetypes = defaultdict(int)
        modes = defaultdict(int)
        dimensions = defaultdict(int)
        for filepath in filepaths:
            get_image_info(filepath, filetypes, modes, dimensions)
        sorted_dimensions = dict(sorted(dimensions.items(), key=lambda item : min(item[0])))

        print(f"{name} total number of files: {sum(filetypes.values())}")
        if get_filetypes:
            print_dictionary(filetypes, root_str=name + " filetypes: ")
        if get_modes:
            print_dictionary(modes, root_str=name + " modes: ")
        if get_dimensions:
            print_dictionary(sorted_dimensions, root_str=name + " sizes: ")
        print('\n')

def main():
    train_filepath=os.path.join( "data", "train")
    test_filepath =os.path.join("data", "test")

    train_clean_filepaths = [os.path.join(train_filepath, "cleanTrain", file) for file in os.listdir(path=os.path.join(train_filepath, "cleanTrain"))]
    test_clean_filepaths = [os.path.join(test_filepath, "cleanTest", file) for file in os.listdir(path=os.path.join(test_filepath, "cleanTest"))]
    train_lsb_filepaths = [os.path.join(train_filepath, "LSBTrain", file) for file in os.listdir(path=os.path.join(train_filepath, "LSBTrain"))]
    test_lsb_filepaths = [os.path.join(test_filepath, "LSBTest", file) for file in os.listdir(path=os.path.join(test_filepath, "LSBTest"))]
    train_pvd_filepaths = [os.path.join(train_filepath, "PVDTrain", file) for file in os.listdir(path=os.path.join(train_filepath, "PVDTrain"))]
    test_pvd_filepaths = [os.path.join(test_filepath, "PVDTest", file) for file in os.listdir(path=os.path.join(test_filepath, "PVDTest"))]

    datasets = {
        "Train clean": train_clean_filepaths,
        "Train LSB": train_lsb_filepaths,
        "Train PVD": train_pvd_filepaths,
        "Test clean": test_clean_filepaths,
        "Test LSB": test_lsb_filepaths,
        "Test PVD": test_pvd_filepaths
    }

    get_file_info(datasets, get_dimensions=False, get_filetypes=False, get_modes=False)

    # print_rgba_values(train_clean_filepaths[0])
    
    
    

if __name__ == "__main__":
    main()