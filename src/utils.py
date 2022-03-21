import json
import random

import numpy as np
import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open('config.json'))
        for k, v in config_args.items():
            setattr(args, k, v)
    del args.config
    return args
