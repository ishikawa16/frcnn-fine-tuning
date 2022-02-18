import random

import numpy as np
import torch


class SaveFeatures:
    def __init__(self):
        self.features = []

    def __call__(self, module, inputs, outputs):
        self.features.append(outputs.clone().detach())

    def clear(self):
        self.outputs = []


def collate_fn(batch):
    return tuple(zip(*batch))


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
