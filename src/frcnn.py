import json
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from alfred import AlfredDataset
from roi_heads import RoIHeads
from two_mlp_head import TwoMLPHead
from utils import collate_fn, fix_seed


class FasterRCNN():
    def __init__(self, args):
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.args = args

    def train_model(self):
        fix_seed(42)

        train_dataset = AlfredDataset('data/alfred-objects', split='train')
        val_dataset = AlfredDataset('data/alfred-objects', split='valid_seen')

        train_dataloader = self.build_dataloader(train_dataset, collate_fn, is_train=True)
        val_dataloader = self.build_dataloader(val_dataset, collate_fn, is_train=False)

        self.load_model()
        self.model.to(self.device)

        optimizer = self.build_optimizer()
        num_epochs = 5

        for epoch in range(num_epochs):
            self.model.train()
            for i, (images, targets) in enumerate(train_dataloader):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if (i+1) % 10 == 0:
                    print(f"Epoch #{epoch+1} Iteration #{i+1} Loss: {loss_value}")

            self.save_model(epoch)

    def predict_oneshot(self):
        self.prepare_model()
        self.model.eval()

        image = Image.open(self.args.image)
        image = transforms.functional.to_tensor(image)
        image = [image.to(self.device)]

        output = self.model(image)
        result = {k: v.tolist() for k, v in output[0].items()}
        with open('output/result.json', 'w') as f:
            json.dump(result, f)

    def prepare_model(self):
        self.load_model()
        if self.args.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint)
        self.model.to(self.device)

    def build_dataloader(self, dataset, collate_fn, is_train):
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

    def build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        return optimizer

    def load_model(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.fix_dimension()
        if self.args.save_features:
            self.fix_roiheads()

    def save_model(self, epoch):
        save_path = f'model/alfred_model_e{epoch+1:02}.pth'
        torch.save(self.model.state_dict(), save_path)

    def fix_dimension(self):
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.args.num_classes)

    def fix_roiheads(self):
        out_features = self.model.roi_heads.box_head.fc7.out_features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        box_head = TwoMLPHead(self.model.backbone.out_channels * box_roi_pool.output_size[0] ** 2, out_features)
        box_predictor = FastRCNNPredictor(in_features, self.args.num_classes)

        self.model.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
            )
