import cv2
import numpy as np
import random
import os
import sys




#TXT_PATH = sys.argv[2]


VERBOSE = True
ENCODE = True
EXTRACT = False



def lsb_hide_rgb(image : str, secret_data: str):
    secret_bin = secret_data
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




def main():
    if len(sys.argv) != 4:
        print("Usage: python t.py <input_dir> <encode_file> <destination_dir>")
        sys.exit(1)

    source_dir = sys.argv[1]
    destination_dir = sys.argv[3]
    message_length = int(sys.argv[2])


    # Get the list of files in the source directory
    source_files = os.listdir(source_dir)
    print(len(source_files))
    for i, file_name in enumerate(source_files):
        src_path = os.path.join(source_dir, file_name)
        file_name = str(message_length)[:-3] + "k_LSB_" + file_name
        dest_path = os.path.join(destination_dir, file_name)
        if(ENCODE):
            image_path = src_path
            image = cv2.imread(image_path) # -> numpy array (w, h, channels)

            # Define secret data to hide
            secret_data = ''.join(str(random.randint(0,1)) for _ in range(message_length * 8))
            # Hide secret data in the image using PVD
            stego_image = lsb_hide_rgb(image, secret_data)

            # Save the stego image
            stego_image_path = dest_path[:-4] + '.png'
            cv2.imwrite(stego_image_path, stego_image)
            print(f"written {message_length} bytes in  {stego_image_path} from {image_path}, number: {i}")

        # Extract the secret data from the stego image
        if(EXTRACT):
            bin, extracted_data = lsb_extract_rgb(stego_image, len(secret_data))



if __name__ == "__main__":

    main()
