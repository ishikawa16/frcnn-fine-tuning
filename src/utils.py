class SaveFeatures:
    def __init__(self):
        self.features = []

    def __call__(self, module, inputs, outputs):
        self.features.append(outputs.clone().detach())

    def clear(self):
        self.outputs = []


def collate_fn(batch):
    return tuple(zip(*batch))
