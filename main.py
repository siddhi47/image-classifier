import torch
import argparse
from image_classifier.train import train, train_args
from image_classifier.test import test, test_args
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--type",
        type=str,
        default="train",
        help="train or test",
    )
    return parser


def main(parser):
    args = parser.parse_args()
    if args.type == "train":
        train(parser)
    elif args.type == "test":
        test(parser)
    else:
        print("Invalid type")


if __name__ == "__main__":
    parser = main_args()
    args = parser.parse_args()
    if args.type == "train":
        parser = train_args(parser)
    elif args.type == "test":
        parser = test_args(parser)
    else:
        print("Invalid type")
    main(parser)
