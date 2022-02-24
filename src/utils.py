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
