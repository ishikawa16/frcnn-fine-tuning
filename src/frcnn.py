import json
import os
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from dataset import ObjectDetectionDataset
from roi_heads import MyRoIHeads
from two_mlp_head import MyTwoMLPHead
from utils import collate_fn, fix_seed


class FasterRCNN():
    def __init__(self, args):
        self.model = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.args = args

    def train_model(self):
        fix_seed(self.args.seed)

        train_dataset = ObjectDetectionDataset(self.args.dataset_dir, self.args.classes, split="train")
        valid_dataset = ObjectDetectionDataset(self.args.dataset_dir, self.args.classes, split="valid")

        train_dataloader = self.build_dataloader(train_dataset, collate_fn, is_train=True)
        valid_dataloader = self.build_dataloader(valid_dataset, collate_fn, is_train=False)

        self.load_model()
        self.model.to(self.device)

        optimizer = self.build_optimizer()

        for epoch in range(self.args.num_epochs):
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

    def test_model(self):
        pass

    def predict_oneshot(self):
        self.prepare_model()
        self.model.eval()

        image = Image.open(self.args.image)
        image = transforms.functional.to_tensor(image)
        image = [image.to(self.device)]

        output = self.model(image)
        output = {k: v.tolist() for k, v in output[0].items()}

        num_objs = len(output["boxes"])
        idx2cls = {idx: cls_ for idx, cls_ in enumerate(self.args.classes)}
        with open(os.path.join(self.args.output_dir, "result.jsonl"), "w") as f:
            for i in range(num_objs):
                if self.args.save_features:
                    result = {
                        "box": output["boxes"][i],
                        "label": idx2cls[output["labels"][i]],
                        "score": output["scores"][i],
                        "fc6_feature": output["fc6_features"][i]
                    }
                else:
                    result = {
                        "box": output["boxes"][i],
                        "label": idx2cls[output["labels"][i]],
                        "score": output["scores"][i]
                    }

                json.dump(result, f)
                f.write("\n")

    def prepare_model(self):
        self.load_model()
        if self.args.checkpoint is not None:
            ckpt = os.path.join(self.args.model_dir, self.args.checkpoint)
            self.model.load_state_dict(torch.load(ckpt))
        self.model.to(self.device)

    def build_dataloader(self, dataset, collate_fn, is_train):
        if is_train:
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.train_batch_size,
                shuffle=True,
                collate_fn=collate_fn
                )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.valid_batch_size,
                shuffle=False,
                collate_fn=collate_fn
                )
        return dataloader

    def build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, **self.args.optim_args)
        return optimizer

    def load_model(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.fix_dimension()
        if self.args.save_features:
            self.fix_roiheads()

    def save_model(self, epoch):
        save_path = os.path.join(self.args.model_dir, f"model_e{epoch+1:02}.pth")
        torch.save(self.model.state_dict(), save_path)

    def fix_dimension(self):
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.args.classes))

    def fix_roiheads(self):
        out_features = self.model.roi_heads.box_head.fc7.out_features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        box_head = MyTwoMLPHead(self.model.backbone.out_channels * box_roi_pool.output_size[0] ** 2, out_features)
        box_predictor = FastRCNNPredictor(in_features, len(self.args.classes))

        self.model.roi_heads = MyRoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.01,
            nms_thresh=0.5,
            detections_per_img=100,
            )
