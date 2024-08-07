from enum import IntEnum
# import ModelTypes enum from model
from model import ModelTypes
from dataclasses import dataclass
import argparse
import torch


@dataclass
class TrainingConfig:
    epochs: int = 2
    learning_rate: float = 0.001
    model_type: ModelTypes = ModelTypes.EfficientNet
    device: str = "default"
    transfer_learning: bool = True
    extract_lsb: bool = False
    batch_size: int = 256
    dataset_types: tuple[str, ...] = ("CLEAN", "LSB")
    step_size: int = 30
    gamma: float = 0.9
    down_sample_size_train: int | None = None
    down_sample_size_test: int | None = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN")
    parser.add_argument(
        "-e", "--epochs", type=int, default=2, help="Number of epochs to train"
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--model-type",
        type=int,
        default=ModelTypes.EfficientNet,
        help="Model type: EfficientNet(1) or ResNet(2) or SWIN(3) or MobileNet(4)",
    )
    parser.add_argument(
        "--device", type=str, default="default", help="Device type: cpu, cuda, or mps"
    )
    parser.add_argument(
        "--transfer-learning",
        action="store_true",
        help="Enable model unrolling and freezing",
    )
    parser.add_argument(
        "--dataset-types",
        type=str,
        nargs="+",
        default=("CLEAN", "LSB"),
        choices=[i.name for i in DatasetTypes],
        help="Dataset type: CLEAN(1), DTC(2), FFT(4), LSB(8), PVD(16), SSB4(32), SSBN(64)",
    )
    parser.add_argument(
        "-el", "--extract-lsb", action="store_true", help="Enable masking bits for LSB"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--step-size", type=int, default=30, help="Specify step size for LR scheduler"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="Specify decay factor for LR scheduler"
    )

    return parser.parse_args()


def get_device(device_argument):
    # if default set automatically
    if device_argument == "default":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    # if argument is not default return user inputted device
    return device_argument


def get_config():
    args = parse_args()
    # Dataset Checks
    assert len(args.dataset_types) > 1, "need more than one dataset"
    for dataset_type in args.dataset_types:
        assert dataset_type in [
            i.name for i in DatasetTypes
        ], f"{dataset_type} is not a valid dataset type"

    # User Input Checks
    assert args.epochs > 0, "# of epochs to train must be positive!"
    assert args.batch_size > 0, "Batch size must be a positive integer!"
    assert args.learning_rate > 0, "LR must be positive!"
    assert 0 < args.gamma < 1, "Gamma must be between 0 and 1!"
    assert args.device in ["cpu", "mps", "cuda", "default"] + [
        f"cuda:{n}" for n in range(8)
    ], "Specified device must be either cpu, cuda, or mps!"
    assert (
        args.model_type == ModelTypes.EfficientNet
        or args.model_type == ModelTypes.ResNet
        or args.model_type == ModelTypes.SWIN
        or args.model_type == ModelTypes.MobileNet
    ), "Model type must either be EfficientNet(1) or ResNet(2)!"

    return TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        device=get_device(args.device),
        transfer_learning=args.transfer_learning,
        extract_lsb=args.extract_lsb,
        batch_size=args.batch_size,
        dataset_types=args.dataset_types,
        step_size=args.step_size,
        gamma=args.gamma,
    )



# Enum to differentiate which dataset to use
class DatasetTypes(IntEnum):
    CLEAN = 1
    DCT = 2
    FFT = 4
    LSB = 8
    PVD = 16
    SSB4 = 32
    SSBN = 64


# since the datset argument takes in a list of strings, this is used to convert that list back to integers for processing later
def enum_names_to_values(names):
    values = []
    for name in names:
        member = DatasetTypes[name]
        values.append(member.value)
    return values
