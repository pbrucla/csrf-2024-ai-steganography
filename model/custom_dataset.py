import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os
from PIL import Image
from dataset import resize_images, extract_lsb_transform
from model import ModelTypes
from main import TrainingConfig

# for determining which dataset to use as specified by the user
import numpy as np

# make a class: Dataloader
class CustomData(Dataset):
    # filepath is the root path for StegoPvd Dataset
    def __init__(
        self,
        extract_lsb,
        dataset_types: list[int],
        mode,
        filepath = os.path.join("data", "CustomStego"),
        color_channel="rgb",
        down_sample_size: None | int = None,
        image_size = 256
    ):
        mode = mode.lower()
        color_channel = color_channel.upper()

        assert (
            down_sample_size >= 1 if down_sample_size is not None else True
        ), f"{down_sample_size} is too small"
        assert mode in ["val", "test", "train"], f"{mode} is not a valid dataset mode"
        assert color_channel in [
            "L",
            "RGB",
            "RGBA",
        ], f"{color_channel} is not a valid color channel"

        self.extract_lsb = extract_lsb

        filepaths = []
        self.class_labels = []

        # file path identification/appendage, filepath contains 7 lists for each set
        for type in dataset_types:
            path_to_folder = os.path.join(filepath, type + ("" if mode == "train" else "test" if mode == "test" else "Val"))
            self.class_labels.append(type)  # put all in labels
            data_classes_paths = [
                os.path.join(path_to_folder, file)
                for file in os.listdir(path=path_to_folder)
            ]
            if down_sample_size is not None:
                data_classes_paths = data_classes_paths[:down_sample_size]

            filepaths.append(data_classes_paths)

        self.all_files = []
        self.dataset_sizes = []
        self.labels = []
        # construct labels and the file dataset
        for n, path in enumerate(filepaths):
            self.all_files.extend(path)
            self.labels.extend([n] * len(path))
            self.dataset_sizes.append(len(path))

        self.labels = torch.tensor(self.labels, dtype=torch.long)

        self.transform = v2.Compose(
            [
                resize_images((image_size, image_size)),
                v2.ToImage(),  # does not scale values
                extract_lsb_transform() if self.extract_lsb else lambda x: x,
                v2.ToDtype(
                    torch.float32
                ),  # preserves original values, no normalize (scale=false default)
                # v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]
        )

    # return length of dataset
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # open in PIL
        filepath = self.all_files[idx]
        # find filepath previosuly
        image = Image.open(filepath)  # directly convert to 32-bit float

        image = self.transform(image)

        # get label
        label = self.labels[idx]

        return image, label


def get_custom_dataset(config):
    train_dataset = CustomData(
        config.extract_lsb,
        config.dataset_types,
        mode="train",
        down_sample_size=config.down_sample_size_train
    )
    test_dataset = CustomData(
        config.extract_lsb,
        config.dataset_types,
        mode="test",
        down_sample_size=config.down_sample_size_test
    )

    return train_dataset, test_dataset

if __name__ == "__main__":

    config = TrainingConfig(
        epochs=10,
        learning_rate=1e-3,
        model_type=ModelTypes.EfficientNet,
        device= "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        transfer_learning=True,
        extract_lsb=False,
        batch_size=256,
        dataset_types=("LSB", "PVD", "Hamming_codes_binary"),
        down_sample_size_train= None,
        down_sample_size_test= None
    )

    get_custom_dataset(config)
