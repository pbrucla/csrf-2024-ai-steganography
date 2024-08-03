import cv2
import numpy as np
import math
import os
import sys
import random


widths = [3,3,4,5,6,7]
ranges = np.array([[0,7],[8,15],[16,31],[32,63],[64,127],[128,255]])
lowers = ranges[:,0]
uppers = ranges[:,1]




VERBOSE = True
ENCODE = True
EXTRACT = False

def f(p1, p2, m):
    ceiling = math.ceil(m/2.0)
    floor = math.floor(m/2.0)
    d = abs(int(p1) - int(p2))
    if (d%2) == 1:
        g1 = p1-ceiling
        g2 = p2+floor
    else:
        g1 = p1-floor
        g2 = p2+ceiling
    return g1, g2




def pvd_hide_rgb(image : str, secret_data: str):
    secret_bin = secret_data
    data_index = 0
    secret_data_len = len(secret_bin)
    
    channels = cv2.split(image)
    for channel in channels:
        rows, cols = channel.shape
        for i in range(rows): # skip a row? maybe change to just 1
            for j in range(0, cols-1, 2): 
                if data_index < secret_data_len:
                    p1 = channel[i, j]
                    p2 = channel[i, j+1]
                    d = int(p2) - int(p1)
                    k = 0
                    for index, element in enumerate(uppers):
                        if abs(d) > element:
                            k = index+1
                    n = widths[k] # number of bits that can be 
                    b = secret_bin[data_index:data_index+n]
                    if len(b) < n:
                        b = b.ljust(n, '0')
                    b = int(b, 2)
                    if d >= 0:
                        dprime = lowers[k] + b
                    else:
                        dprime = -(lowers[k]+b)
                    test = f(p1, p2, uppers[k] - d)
                    if(test[0] > 255 or test[1] > 255):
                        continue
                    data_index += n
                    m = dprime - d
                    g1, g2 = f(p1, p2, m)
                    channel[i, j] = g1
                    channel[i, j+1] = g2
                    if data_index >= secret_data_len:
                        break


    return cv2.merge(channels)

def pvd_extract_rgb(stego_image, secret_len):
    channels = cv2.split(stego_image)
    extracted_bin = ''
    
    for channel in channels:
        rows, cols = channel.shape
        for i in range(rows):
            for j in range(0, cols-1, 2):
                if len(extracted_bin) < secret_len * 8:
                    p1 = channel[i, j]
                    p2 = channel[i, j+1]
                    d = abs(int(p1) - int(p2))
                    k = 0
                    for index, element in enumerate(uppers):
                        if d > element:
                            k = index+1
                    test = f(p1, p2, uppers[k] - d)
                    if(test[0] > 255 or test[1] > 255):
                        continue
                    b = abs(d) - lowers[k]
                    b = format(b, '08b')[-widths[k]:]
                    extracted_bin += b

    
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
        file_name = str(message_length)[:-3] + "k_PVD_" + file_name
        dest_path = os.path.join(destination_dir, file_name)
        if(ENCODE):
            image_path = src_path
            image = cv2.imread(image_path) # -> numpy array (w, h, channels)

            # Define secret data to hide
            secret_data = ''.join(str(random.randint(0,1)) for _ in range(message_length * 8))
            # Hide secret data in the image using PVD
            stego_image = pvd_hide_rgb(image, secret_data)

            # Save the stego image
            stego_image_path = dest_path
            cv2.imwrite(stego_image_path, stego_image)
            print(f"written {message_length} bytes in {stego_image_path} from {image_path}, number: {i}")

        # Extract the secret data from the stego image
        if(EXTRACT):
            bin, extracted_data = pvd_extract_rgb(stego_image, len(secret_data))



if __name__ == "__main__":
    main()