import argparse

from frcnn import FasterRCNN
from utils import parse_with_config


def main(opts):
    frcnn = FasterRCNN(opts)

    if opts.mode == "train":
        frcnn.train_model()

    elif opts.mode == "test":
        frcnn.prepare_model()
        frcnn.model.eval()
        frcnn.test_model()

    elif opts.mode == "predict_oneshot":
        frcnn.prepare_model()
        frcnn.model.eval()
        frcnn.predict_oneshot()

    else:
        raise ValueError("Invalid argument")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--image", type=str)
    parser.add_argument("--save_features", action="store_true")

    args = parse_with_config(parser)

    main(args)
