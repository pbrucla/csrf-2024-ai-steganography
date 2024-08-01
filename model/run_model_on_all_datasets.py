import csv
import torch
from pathlib import Path

from main import train_model, TrainingConfig
from model import ModelTypes
from config import DatasetTypes
from custom_dataset import get_custom_dataset


if __name__ == "__main__":

    use_custom_dataset = True

    if use_custom_dataset:
        for model in ModelTypes:
            directory = Path("temp")
            directory.mkdir(parents=True, exist_ok=True)
            filename = directory / f"{model}_accuracies.csv"

            stego_images = [
                "LSB", "PVD", "Hamming_codes_binary"
            ]

            stego_accuracy = ["Accuracy"]
            stego_f1 = [["Class 1"],["F1"],["Class 2"],["F1"]]

            for i in stego_images:
                config = TrainingConfig(
                    epochs=10,
                    learning_rate=1e-3,
                    model_type=model,
                    device= "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                    transfer_learning=True,
                    extract_lsb=False,
                    batch_size=256,
                    dataset_types=("clean", i),
                    down_sample_size_train= None,
                    down_sample_size_test= None
                )
                train_dataset, test_dataset = get_custom_dataset(config)
                test_statistics, class_names = train_model(config, train_dataset, test_dataset)
                stego_accuracy.append(test_statistics[0])
                stego_f1[0].append(class_names[0])
                stego_f1[1].append(test_statistics[1][0])
                stego_f1[2].append(class_names[1])
                stego_f1[3].append(test_statistics[1][1])

            config.dataset_types = stego_images + ["clean"]
            all_datasets_statistics = train_model(config)
            all_class_names = all_datasets_statistics[1]
            all_f1 = list(all_datasets_statistics[0][1])
            all_f1.insert(0, "F1")
            all_class_names.insert(0, "Classes:")

            stego_images.insert(0, "Clean + ")

            with open(filename, "w") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(stego_images)
                csvwriter.writerow(stego_accuracy)
                csvwriter.writerows(stego_f1)
                for i in range(2):
                    csvwriter.writerow([""])
                csvwriter.writerow(["All datsets:"])
                csvwriter.writerow(["Total Accuracy:", all_datasets_statistics[0][0]])
                csvwriter.writerow(all_class_names)
                csvwriter.writerow(all_f1)



    if not use_custom_dataset:
        directory = Path("temp")
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / "stego_accuracies.csv"

        non_rgb_datasets = ["SSB4", "SSBN"]
        stego_images = [
            i.name
            for i in DatasetTypes
            if i.name != "CLEAN" and not i.name in non_rgb_datasets
        ]

        stego_accuracy = ["Accuracy"]
        stego_f1 = [["Class 1"],["F1"],["Class 2"],["F1"]]

        for i in stego_images:
            config = TrainingConfig(
                epochs=10,
                learning_rate=1e-3,
                model_type=ModelTypes.EfficientNet,
                device= "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                transfer_learning=True,
                extract_lsb=False,
                batch_size=256,
                dataset_types=("CLEAN", i),
                down_sample_size_train= None,
                down_sample_size_test= None
            )
            test_statistics, class_names = train_model(config)
            stego_accuracy.append(test_statistics[0])
            stego_f1[0].append(class_names[0])
            stego_f1[1].append(test_statistics[1][0])
            stego_f1[2].append(class_names[1])
            stego_f1[3].append(test_statistics[1][1])

        config.dataset_types = stego_images + ["CLEAN"]
        all_datasets_statistics = train_model(config)
        all_class_names = all_datasets_statistics[1]
        all_f1 = list(all_datasets_statistics[0][1])
        all_f1.insert(0, "F1")
        all_class_names.insert(0, "Classes:")

        stego_images.insert(0, "Clean + ")

        with open(filename, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(stego_images)
            csvwriter.writerow(stego_accuracy)
            csvwriter.writerows(stego_f1)
            for i in range(2):
                csvwriter.writerow([""])
            csvwriter.writerow(["All datsets:"])
            csvwriter.writerow(["Total Accuracy:", all_datasets_statistics[0][0]])
            csvwriter.writerow(all_class_names)
            csvwriter.writerow(all_f1)
