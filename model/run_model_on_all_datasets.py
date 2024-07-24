import csv

from main import train_model, TrainingConfig
from model import ModelTypes
from config import DatasetTypes

if __name__ == "__main__":
    filename = "temp/stego_accuracies.csv"
    non_rgb_datasets = ["SSB4", "SSBN"]
    stego_images = [
        i.name
        for i in DatasetTypes
        if i.name != "CLEAN" and not i.name in non_rgb_datasets
    ]

    stego_accuracy = []

    for i in stego_images:
        config = TrainingConfig(
            epochs=10,
            learning_rate=1e-3,
            model_type=ModelTypes.EfficientNet,
            device="default",
            transfer_learning=True,
            extract_lsb=False,
            batch_size=256,
            dataset_types=("CLEAN", i),
        )
        stego_accuracy.append(train_model(config)[-1])

    config.dataset_types = stego_images + ["CLEAN"]
    all_datasets_accuracy = train_model(config)[-1]

    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(stego_images)
        csvwriter.writerow(stego_accuracy)
        csvwriter.writerow([])
        csvwriter.writerow(["all datasets"])
        csvwriter.writerow([all_datasets_accuracy])
