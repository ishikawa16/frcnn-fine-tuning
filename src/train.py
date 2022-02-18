import torch
from torch.utils.data import DataLoader, random_split

from alfred import AlfredDataset
from frcnn import FasterRCNN
from utils import collate_fn


def build_dataloader(dataset, collate_fn, is_train):
    if is_train:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
            )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_fn
            )

    return dataloader


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = AlfredDataset('data/alfred_pick_only')
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = build_dataloader(train_dataset, collate_fn, is_train=True)
    val_dataloader = build_dataloader(val_dataset, collate_fn, is_train=False)

    model = FasterRCNN(num_classes=2, training=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(train_dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f"Epoch #{epoch+1} Iteration #{i+1} Loss: {loss_value}")

        torch.save(model.state_dict(), f'model/alfred_model_e{epoch+1:02}.pth')


if __name__=='__main__':
    main()
