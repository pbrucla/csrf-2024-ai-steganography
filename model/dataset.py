import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os
from PIL import Image
#for determining which dataset to use as specified by the user
import numpy as np

from config import DatasetTypes

class resize_images(object):
    def __init__(self, target_size=(128,128)):
        self.target_size = target_size

    def __call__(self, img):
        """
        :param img: (PIL): Image 
        """
        padding = (
            max(0, (self.target_size[0] - img.height) // 2),
            max(0, (self.target_size[1] - img.width) // 2),
            max(0, (self.target_size[0] - img.height + 1) // 2),
            max(0, (self.target_size[1] - img.width + 1) // 2),
        )
        transform = v2.Compose([
            v2.Pad(padding=padding, fill=0, padding_mode='constant'),
            v2.CenterCrop(self.target_size)
        ])
        return transform(img)
    
    def __repr__(self):
        return self.__class__.__name__ + '(target_size={})'.format(self.target_size)
    

    
# make a class: Dataloader
class Data(Dataset):
    # filepath is the root path for StegoPvd Dataset
    def __init__(self, extract_lsb, dataset_types, filepath, clean_path="cleanTrain", dct_path="DCTTrain", fft_path = "FFTTrain", lsb_path = "LSBTrain", pvd_path="PVDTrain", ssb4_path = "SSB4Train", ssbn_path = "SSBNTrain"): 
        
        self.extract_lsb = extract_lsb
        
        user_options = 0
        for type in dataset_types:
            user_options |= type
        
        clean_filepaths = []
        stego_filepaths = []

        if user_options & DatasetTypes.CLEAN:
            clean_filepaths.extend([os.path.join(filepath, clean_path, file) for file in os.listdir(path=os.path.join(filepath, clean_path))])
        elif user_options & DatasetTypes.DCT:
            stego_filepaths.extend([os.path.join(filepath, dct_path, file) for file in os.listdir(path=os.path.join(filepath, dct_path))])
        elif user_options & DatasetTypes.FFT:
            stego_filepaths.extend([os.path.join(filepath, fft_path, file) for file in os.listdir(path=os.path.join(filepath, fft_path))])
        elif user_options & DatasetTypes.LSB:
            stego_filepaths.extend([os.path.join(filepath, lsb_path, file) for file in os.listdir(path=os.path.join(filepath, lsb_path))])
        elif user_options & DatasetTypes.PVD:
            stego_filepaths.extend([os.path.join(filepath, pvd_path, file) for file in os.listdir(path=os.path.join(filepath, pvd_path))])
        elif user_options & DatasetTypes.SSB4:
            stego_filepaths.extend([os.path.join(filepath, ssb4_path, file) for file in os.listdir(path=os.path.join(filepath, ssb4_path))])
        elif user_options & DatasetTypes.SSBN:
            stego_filepaths.extend([os.path.join(filepath, ssbn_path, file) for file in os.listdir(path=os.path.join(filepath, ssbn_path))])

        self.all_files = clean_filepaths + stego_filepaths
        self.labels = [0] * len(clean_filepaths) + [1] * len(stego_filepaths)

        self.transform = v2.Compose([
            resize_images((128, 128)),
            v2.ToTensor(), #does not scale values
            v2.ToDtype(torch.float32), #preserves original values, no normalize (scale=false default)
            #v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    #return length of dataset
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # open in PIL
        filepath = self.all_files[idx]
        # find filepath previosuly
        image = Image.open(filepath) #directly convert to 32-bit float

        image = self.transform(image)

        if self.extract_lsb:
            image = image & 1
    
        #get label
        label = self.labels[idx]

        return image, label
        