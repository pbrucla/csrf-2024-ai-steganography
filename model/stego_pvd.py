import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os
from PIL import Image

# make a class: Dataloader
class StegoPvd(Dataset):
    # filepath is the root path for StegoPvd Dataset
    def __init__(self, filepath, clean_path="cleanTrain", stego_path="stegoTrain"):
        clean_filepaths = [os.path.join(filepath, clean_path, file) for file in os.listdir(path=os.path.join(filepath, clean_path))]
        stego_filepaths = [os.path.join(filepath, stego_path, file) for file in os.listdir(path=os.path.join(filepath, stego_path))]
        self.all_files = clean_filepaths + stego_filepaths
        self.labels = [0] * len(clean_filepaths) + [1] * len(stego_filepaths)

        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        self.transform = v2.Compose([
            v2.Resize((128, 128)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    #return length of dataset
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # open in PIL
        filepath = self.all_files[idx]
        # find filepath previosuly
        image = Image.open(filepath).convert('RGB')
        image = self.transform(image)
        
        #get label
        label = self.labels[idx]

        return image, label
        