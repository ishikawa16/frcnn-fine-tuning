import argparse

from frcnn import FasterRCNN
from utils import parse_with_config


def main(args):
    frcnn = FasterRCNN(args)
    if args.mode == "train":
        frcnn.train_model()
    elif args.mode == "test":
        frcnn.test_model()
    elif args.mode == "predict_oneshot":
        frcnn.predict_oneshot()
    else:
        raise OSError("Invalid argument")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image")

    args = parse_with_config(parser)

    main(args)
