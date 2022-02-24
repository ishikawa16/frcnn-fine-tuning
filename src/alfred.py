import json
import os

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AlfredDataset(Dataset):
    def __init__(self, root, split):
        super().__init__()
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'image'))))
        with open(os.path.join(root, f'{split}.jsonl')) as f:
            self.data = sorted([json.loads(line) for line in f], key=lambda x:x['id'])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'image', self.imgs[idx])
        img = Image.open(img_path)
        img = transforms.functional.to_tensor(img)

        data = self.data[idx]

        boxes = [data['target']['box']]
        num_objs = 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.data)
