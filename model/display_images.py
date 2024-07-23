# for load random image
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def load_random_images(directory, num_images):
    images = []
    image_names = os.listdir(directory)
    random_images = random.sample(image_names, num_images)
    for image_name in random_images:
        image_path = os.path.join(directory, image_name)
        images.append(Image.open(image_path))
    return images

def display_image(title, images, num_rows, num_cols):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15,15))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()

# Loads random clean and stego image (from data/train/cleanTrain and data/train/LSBTrain)
def display_random_images():
    clean_directory = os.path.join("data", "train", "cleanTrain")
    dct_directory = os.path.join("data", "train", "DCTTrain")
    fft_directory = os.path.join("data", "train", "FFTTrain")
    lsb_directory = os.path.join("data", "train", "LSBTrain")
    pvd_directory = os.path.join("data", "train", "PVDTrain")
    ssb4_directory = os.path.join("data", "train", "SSB4Train")
    ssbn_directory = os.path.join("data", "train", "SSBNTrain")

    
    clean_images = load_random_images(clean_directory, 64)
    dct_images = load_random_images(dct_directory, 64)
    fft_images = load_random_images(fft_directory, 64)
    lsb_images = load_random_images(lsb_directory, 64)
    pvd_images = load_random_images(pvd_directory, 64)
    ssb4_images = load_random_images(ssb4_directory, 64)
    ssbn_images = load_random_images(ssbn_directory, 64)
    
    display_image("Clean images", clean_images, 8, 8)
    display_image("DCT images", dct_images, 8, 8)
    display_image("FFT images", fft_images, 8, 8)
    display_image("LSB images", lsb_images, 8, 8)
    display_image("PVD images", pvd_images, 8, 8)
    display_image("SSB4 images", ssb4_images, 8, 8)
    display_image("SSBN images", ssbn_images, 8, 8)

# Example usage
if __name__ == "__main__":
   print("Displaying images")
   display_random_images()
   print("Finished")