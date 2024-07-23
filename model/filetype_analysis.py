from PIL import Image
from collections import defaultdict
import os   
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_image_info(image_path, filetype_dictionary, mode_dictionary, size_dictionary):
    if Path(image_path).suffix.lower() == ".txt":
        print(f"\nBad filepath: {image_path}\n")
        return
    if os.path.isdir(os.path.join(image_path)):
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
    print()

def print_rgba_values(image_path: str): # Probably change to show more clearly
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
                
def print_dimensions_bar_chart(dimensions_dictionary: dict, dataset_name: str):
    widths = [dimensions[0] for dimensions in dimensions_dictionary.keys()]
    heights = [dimensions[1] for dimensions in dimensions_dictionary.keys()]
    weights = list(dimensions_dictionary.values())

    print(f'width_dist len: {len(widths)}')
    print(f'height_dist len: {len(heights)}')

    if len(widths) != len(heights) or len(widths) != len(weights):
        print(f'''ERROR: Couldn\'t print scatterplot {dataset_name} due to dimension mismatch.
              dist1 size: {len(widths)}
              dist2 size: {len(heights)}
              weights size: {len(weights)}''')
        return
    
    z = np.zeros_like(widths)
    dx = dy = np.ones_like(widths) * 3
    weights_log = np.log(weights)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.bar3d(widths, heights, z, dx, dy, weights_log, color='b')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Log(occurences)')
    ax.set_title('Log of Number of Occurences of Image Dimensions')
    plt.draw()
    filepath = os.path.join('temp', 'plot.png')
    print(f'Saving plot to {filepath}')
    plt.savefig('train_clean_plot.png')
    plt.close(fig)

def get_dataset_info(name: str, filepaths: str, print_filetypes=True, print_modes=True, print_dimensions=True):
    filetypes = defaultdict(int)
    modes = defaultdict(int)
    dimensions = defaultdict(int)
    for filepath in filepaths:
        get_image_info(filepath, filetypes, modes, dimensions)
    sorted_dimensions = dict(sorted(dimensions.items(), key=lambda item : min(item[0])))

    print(f"{name} total number of files: {sum(filetypes.values())}")
    if print_filetypes:
        print_dictionary(filetypes, root_str=name + " filetypes: ")
    if print_modes:
        print_dictionary(modes, root_str=name + " modes: ")
    if print_dimensions:
        print_dictionary(sorted_dimensions, root_str=name + " sizes: ")
    print()

    return filetypes, modes, sorted_dimensions

def get_filepaths(base_filepath, folder):
    return [os.path.join(base_filepath, folder, file) for file in os.listdir(path=os.path.join(base_filepath, folder))]


def main():
    train_filepath = os.path.join("data", "train")
    test_filepath = os.path.join("data", "test")

    train_clean_filepaths = get_filepaths(train_filepath, "cleanTrain")
    test_clean_filepaths = get_filepaths(test_filepath, "cleanTest")
    train_lsb_filepaths = get_filepaths(train_filepath, "LSBTrain")
    test_lsb_filepaths = get_filepaths(test_filepath, "LSBTest")
    train_pvd_filepaths = get_filepaths(train_filepath, "PVDTrain")
    test_pvd_filepaths = get_filepaths(test_filepath, "PVDTest")
    train_dct_filepaths = get_filepaths(train_filepath, "DCTTrain")
    test_dct_filepaths = get_filepaths(test_filepath, "DCTTest")
    train_fft_filepaths = get_filepaths(train_filepath, "FFTTrain")
    test_fft_filepaths = get_filepaths(test_filepath, "FFTTest")
    train_ssb4_filepaths = get_filepaths(train_filepath, "SSB4Train")
    test_ssb4_filepaths = get_filepaths(test_filepath, "SSB4Test")
    train_ssbn_filepaths = get_filepaths(train_filepath, "SSBNTrain")
    test_ssbn_filepaths = get_filepaths(test_filepath, "SSBNTest")

    datasets = {
        "Train clean": train_clean_filepaths,
        "Test clean": test_clean_filepaths,
        "Train LSB": train_lsb_filepaths,
        "Test LSB": test_lsb_filepaths,
        "Train PVD": train_pvd_filepaths,
        "Test PVD": test_pvd_filepaths,
        "Train DCT": train_dct_filepaths,
        "Test DCT": test_dct_filepaths,
        "Train FFT": train_fft_filepaths,
        "Test FFT": test_fft_filepaths,
        "Train SSB4": train_ssb4_filepaths,
        "Test SSB4": test_ssb4_filepaths,
        "Train SSBN": train_ssbn_filepaths,
        "Test SSBN": test_ssbn_filepaths
    }

    get_dataset_info(name="nonrgb train", filepaths=get_filepaths(os.path.join('data', 'train', 'cleanTrain'), folder='nonrgb'), print_dimensions=False)
    get_dataset_info(name="nonrgb test", filepaths=get_filepaths(os.path.join('data', 'train', 'cleanTest'), folder='nonrgb'), print_dimensions=False)

    # for name, filepaths in datasets.items():
    #     get_dataset_info(name=name, filepaths=filepaths, print_dimensions=False, print_filetypes=True, print_modes=True)
    
    # filetypes, modes, dimensions = get_dataset_info(name="Train clean", filepaths=train_clean_filepaths, print_dimensions=False, print_filetypes=True, print_modes=True)
    # filetypes, modes, dimensions = get_dataset_info(name="Test clean", filepaths=test_clean_filepaths, print_dimensions=False, print_filetypes=True, print_modes=True)

    # print_rgba_values(train_clean_filepaths[0])
    # print_dimensions_bar_chart(dimensions_dictionary=dimensions, dataset_name="Train clean")
    # print_dictionary(dimensions, root_str='Train clean dimensions: ')
    
    
    

if __name__ == "__main__":
    main()