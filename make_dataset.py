import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from lib import pose_detection
import argparse
import pathlib
import numpy as  np

parser = argparse.ArgumentParser(description="Train LSTM model using specified dataset")
parser.add_argument("-q", "--quiet", action="store_true", help="hide progress on each video")
parser.add_argument("-s", "--show", action="store_true", help="show video being processed")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--train", action="store_true", help="make training dataset")
group.add_argument("-v", "--val", action="store_true", help="make validation dataset")
args = parser.parse_args()

if __name__ == "__main__":
    if args.train:
        path = str(pathlib.Path().resolve()) + "\\video\\train"
    elif args.val:
        path = str(pathlib.Path().resolve()) + "\\video\\validation"

    classes = os.listdir(path)

    print("MAKING DATASET, PLEASE WAIT!")

    x, y = pose_detection.get_coordinates(classes, path, args.quiet, args.show)

    print("DATASET COMPLETE")

    if args.train:
        np.save(os.fspath(pathlib.Path(__file__).parent.parent / "dataset/train_x"), x)
        np.save(os.fspath(pathlib.Path(__file__).parent.parent / "dataset/train_y"), y)
    elif args.val:   
        np.save(os.fspath(pathlib.Path(__file__).parent.parent / "dataset/val_x"), x)
        np.save(os.fspath(pathlib.Path(__file__).parent.parent / "dataset/val_y"), y) 