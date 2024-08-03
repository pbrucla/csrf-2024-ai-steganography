from PIL import Image
import os


def main():
    clean_train_filepath = os.path.join("data", "train", "cleanTrain")
    clean_test_filepath = os.path.join("data", "test", "cleanTest")

    # remove_nonrgb(clean_train_filepath)
    # remove_nonrgb(clean_test_filepath)
    remove_nonrgb()


def remove_nonrgb(filepath):
    filenames = os.listdir(filepath)
    print(f"{len(filenames)} files in {filepath}")

    newpath = os.path.join(filepath, "nonrgb")
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    count = 0

    for filename in filenames:
        if os.path.isdir(os.path.join(filepath, filename)):
            continue
        with Image.open(os.path.join(filepath, filename)) as img:
            if img.mode != "RGB":
                os.rename(
                    os.path.join(filepath, filename), os.path.join(newpath, filename)
                )
                count += 1

    print(f"Moved {count} files from {filepath} to {newpath}")


if __name__ == "__main__":
    # remove all directories with non-rgb using
    # find . -type d -name "nonrgb" -exec rm -rf {} +
    main()
