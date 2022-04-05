import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import TwoMLPHead


class MyTwoMLPHead(TwoMLPHead):
    def __init__(self, in_channels, representation_size):
        super().__init__(in_channels, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        fc6_features = x
        x = F.relu(self.fc7(x))

        return x, fc6_features
