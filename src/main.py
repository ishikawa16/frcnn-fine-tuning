import argparse

from frcnn import FasterRCNN


def main(args):
    frcnn = FasterRCNN(args)
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

    parser.add_argument('--checkpoint')
    parser.add_argument('--mode', required=True)
    parser.add_argument('--num_classes', required=True, type=int)
    parser.add_argument('--save_features', action='store_true')

    args = parser.parse_args()

    main(args)
