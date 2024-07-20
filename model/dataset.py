import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os
from PIL import Image
#for determining which dataset to use as specified by the user
import numpy as np

from config import DatasetTypes

def accuracy_metric(predictions, labels):
    predicted_classes = torch.argmax(predictions, dim=-1)
    correct_predictions = (predicted_classes == labels).sum().item()
    
    return correct_predictions, labels.size(0)

#returns a list of class accuracies
def equal_accuracy_metric(predictions, labels): 
    accuracy_list = []
    for dataset in datsets:
        accuracy_list.append(accuracy_metric(dataset.predictions, dataset.labels))
    return accuracy_list

        
 
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
    

class extract_lsb_transform(object):
    def __call__(self, tensor):
        return tensor & 1
    def __repr__(self):
        return self.__class__.__name__
    
# make a class: Dataloader
class Data(Dataset):
    # filepath is the root path for StegoPvd Dataset
    def __init__(self, extract_lsb, dataset_types: list[int], filepath, mode): 
        
        assert mode in ["val", "test", "train"], f"{mode} is not a valid dataset mode"

        self.extract_lsb = extract_lsb
        
        # moving optional parameters to map
        path_to_folder = {
            DatasetTypes.CLEAN: "clean",
            DatasetTypes.DCT: "DCT",
            DatasetTypes.FFT: "FFT",
            DatasetTypes.LSB: "LSB",
            DatasetTypes.PVD: "PVD",
            DatasetTypes.SSB4: "SSB4",
            DatasetTypes.SSBN: "SSBN"
        }

        filepaths = [] 
        self.class_labels = []
    
        #file path identification/appendage, filepath contains 7 lists for each set
        for type in dataset_types:
            self.class_labels.append(path_to_folder[type]) #put all in labels
            folder = path_to_folder.get(type) + mode.capitalize()
            filepaths.append([os.path.join(filepath, folder, file) for file in os.listdir(path=os.path.join(filepath, folder))])
               
        self.all_files = []
        self.dataset_sizes = []
        for n, path in enumerate(filepaths):
            self.all_files.extend(path)
            self.labels.extend([n] * len(path)) 
            self.dataset_sizes.append(len(path))
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        self.transform = v2.Compose([
            resize_images((128, 128)),
            v2.ToImage(), #does not scale values
            extract_lsb_transform() if self.extract_lsb else lambda x: x,
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
        image = Image.open(filepath).convert("RGB") #directly convert to 32-bit float

        image = self.transform(image)
    
        #get label
        label = self.labels[idx]

        return image, label
        