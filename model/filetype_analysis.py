from PIL import Image
from collections import defaultdict
import os
import numpy as np

def get_image_info(image_path, filetype_dictionary, mode_dictionary):
    with Image.open(image_path) as img:
        filetype = img.format
        filetype_dictionary[filetype] += 1
        mode = img.mode
        mode_dictionary[mode] += 1
        
def update_dictionary(filepaths, filetype_dictionary, mode_dictionary):
    for path in filepaths:
        get_image_info(path, filetype_dictionary, mode_dictionary)

def print_dictionary(dict: ct[str, int], root_str : str) -> None:
    print(root_str, end=' ')
    for key, val in dict.items():
        print(key, ": ", val, end=' ')
    print()

def print_alpha_values(image_path):
    with Image.open(image_path) as img:
        if (img.mode == 'RGBA'):
            red = img.getchannel('R')
            green = img.getchannel('G')
            blue = img.getchannel('B')
            alpha = img.getchannel('A')
            for y in range(alpha.height):
                for x in range(alpha.width):
                    print(f"({red.getpixel((x, y))}, {green.getpixel((x, y))}, {blue}, end=' ')
, {}alpha                print()
                print()

def get_filetype_mode_info(train_clean_filepaths, train_stego_filepaths, test_clean_filepaths, test_stego_filepaths):
    train_clean_filetypes = defaultdict(int)
    train_clean_modes = defaultdict(int)
    update_dictionary(train_clean_filepaths, train_clean_filetypes, train_clean_modes)

    train_stego_filetypes = defaultdict(int)
    train_stego_modes = defaultdict(int)
    update_dictionary(train_stego_filepaths, train_stego_filetypes, train_stego_modes)

    test_clean_filetypes = defaultdict(int)
    test_clean_modes = defaultdict(int)
    update_dictionary(test_clean_filepaths, test_clean_filetypes, test_clean_modes)

    test_stego_filetypes = defaultdict(int)
    test_stego_modes = defaultdict(int)
    update_dictionary(test_stego_filepaths, test_stego_filetypes, test_stego_modes)
    
    print_dictionary(train_clean_filetypes, root_str="Train clean filetypes: ")
    print_dictionary(train_stego_filetypes, root_str="Train stego filetypes: ")
    print_dictionary(train_clean_modes, root_str="Train clean modes: ")
    print_dictionary(train_stego_modes, root_str="Train stego modes: ")
    print_dictionary(test_clean_filetypes, root_str="Test clean filetypes: ")
    print_dictionary(test_stego_filetypes, root_str="Test stego filetypes: ")
    print_dictionary(test_clean_modes, root_str="Test clean modes: ")
    print_dictionary(test_stego_modes, root_str="Test stego modes: ")

def main():
    train_filepath=os.path.join( "Stego-pvd-dataset", "train")
    test_filepath =os.path.join("Stego-pvd-dataset", "test")

    train_clean_filepaths = [os.path.join(train_filepath, "cleanTrain", file) for file in os.listdir(path=os.path.join(train_filepath, "cleanTrain"))]
    train_stego_filepaths = [os.path.join(train_filepath, "stegoTrain", file) for file in os.listdir(path=os.path.join(train_filepath, "stegoTrain"))]
    test_clean_filepaths = [os.path.join(test_filepath, "cleanTest", file) for file in os.listdir(path=os.path.join(test_filepath, "cleanTest"))]
    test_stego_filepaths = [os.path.join(test_filepath, "stegoTest", file) for file in os.listdir(path=os.path.join(test_filepath, "stegoTest"))]

    # get_filetype_mode_info(train_clean_filepaths, train_stego_filepaths, test_clean_filepaths, test_stego_filepaths)

    print_alpha_values(train_clean_filepaths[0])
    
    
    

if __name__ == "__main__":
    main()