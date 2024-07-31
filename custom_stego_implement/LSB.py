import cv2
import numpy as np
import math
import os
import sys




#TXT_PATH = sys.argv[2]


VERBOSE = False
ENCODE = True
EXTRACT = False



def lsb_hide_rgb(image : str, secret_data):
    secret_bin = ''.join(format(ord(i), '08b') for i in secret_data)
    data_index = 0
    secret_data_len = len(secret_bin)
    
    channels = cv2.split(image)
    for channel in channels:
        rows, cols = channel.shape
        for i in range(rows): # skip a row? maybe change to just 1
            for j in range(cols): 
                if data_index < secret_data_len:
                    p1 = channel[i, j]
                    p1 = p1 & 0xFE
                    p1 = p1 | int(secret_bin[data_index],2)
                    channel[i, j] = p1
                    data_index += 1
    return cv2.merge(channels)

def lsb_extract_rgb(stego_image, secret_len):
    channels = cv2.split(stego_image)
    extracted_bin = ''
    
    for channel in channels:
        rows, cols = channel.shape
        for i in range(rows):
            for j in range(cols):
                if len(extracted_bin) < secret_len * 8:
                    p1 = channel[i, j]
                    extracted_bin += str(p1 & 1)
    secret_data = ''.join(chr(int(extracted_bin[i:i+8], 2)) for i in range(0, len(extracted_bin), 8))
    return extracted_bin, secret_data[:secret_len]

# Load an example image



def binary_to_string(binary_str):
    chars = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
    ascii_str = ''.join(chr(int(char, 2)) for char in chars)
    return ascii_str


if(VERBOSE):
    print(binary_to_string(bin))


def main():
    if len(sys.argv) != 4:
        print("Usage: python t.py <input_dir> <encode_file> <destination_dir>")
        sys.exit(1)

    source_dir = sys.argv[1]
    destination_dir = sys.argv[3]


    # Get the list of files in the source directory
    source_files = os.listdir(source_dir)
    count = 0
    print(len(source_files))
    for i, file_name in enumerate(source_files):
        src_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(destination_dir, file_name)
        if(ENCODE):
            count += 1
            image_path = src_path
            image = cv2.imread(image_path) # -> numpy array (w, h, channels)

            # Define secret data to hide
            with open(TXT_PATH, 'r') as text:
                secret_data = text.read()
                secret_data.replace('\n', '')
            # Hide secret data in the image using PVD
            stego_image = lsb_hide_rgb(image, secret_data)

            # Save the stego image
            stego_image_path = dest_path
            cv2.imwrite(stego_image_path, stego_image)
            print(f"written: {stego_image_path} from {image_path}, number: {count}")

        # Extract the secret data from the stego image
        if(EXTRACT):
            bin, extracted_data = lsb_extract_rgb(stego_image, len(secret_data))



#if __name__ == "__main__":
    #main()
InputImage = sys.argv[1]
textFile = sys.argv[2]
OutputImage = sys.argv[3]
image = cv2.imread(InputImage)

with open(textFile, 'r') as text:
    secret_data = text.read()
    secret_data.replace('\n', '')

stego_image = lsb_hide_rgb(image, secret_data)

if(VERBOSE):
    print(f"hid {textFile} in {OutputImage} from {InputImage}")


cv2.imwrite(OutputImage, stego_image)

if(EXTRACT):
    bin, extracted_data = lsb_extract_rgb(stego_image, len(secret_data))
    print(binary_to_string(bin))
