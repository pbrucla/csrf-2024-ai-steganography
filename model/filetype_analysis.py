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
    plt.savefig(os.path.join('temp', 'plot.png'), dpi=300, bbox_inches='tight')
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
    print('\n')

    return filetypes, modes, sorted_dimensions

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

    filetypes, modes, dimensions = get_dataset_info(name="Train clean", filepaths=train_clean_filepaths, print_dimensions=False, print_filetypes=False, print_modes=False)

    # print_rgba_values(train_clean_filepaths[0])
    print_dimensions_bar_chart(dimensions_dictionary=dimensions, dataset_name="Train clean")
    # print_dictionary(dimensions, root_str='Train clean dimensions: ')
    
    
    

if __name__ == "__main__":
    main()