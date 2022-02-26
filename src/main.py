import argparse

from frcnn import FasterRCNN


def main(args):
    frcnn = FasterRCNN(num_classes=2)
    if args.mode == 'train':
        frcnn.train_model()
    elif args.mode == 'test':
        frcnn.test_model()
    elif args.mode == 'predict_oneshot':
        frcnn.predict_oneshot()
    else:
        raise OSError('Invalid argument')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode')

    args = parser.parse_args()

    main(args)
