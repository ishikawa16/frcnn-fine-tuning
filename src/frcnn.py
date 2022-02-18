from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from roi_heads import RoIHeads
from utils import SaveFeatures


class FasterRCNN(nn.Module):
    def __init__(self, num_classes, training, checkpoint=None, save_features=False, coco_pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.training = training

        self.model = fasterrcnn_resnet50_fpn(pretrained=coco_pretrained)

        self.fix_dimension()
        if save_features:
            self.fix_roiheads()
            self.register_hook()

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint)

    def fix_dimension(self):
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

    def fix_roiheads(self):
        out_features = self.model.roi_heads.box_head.fc7.out_features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        box_head = TwoMLPHead(self.model.backbone.out_channels * box_roi_pool.output_size[0] ** 2, out_features)
        box_predictor = FastRCNNPredictor(in_features, self.num_classes)

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

    def register_hook(self):
        save_features = SaveFeatures()
        self.model.roi_heads.box_head.fc6.register_forward_hook(save_features)

    def forward(self, images, targets=None):
        if self.training:
            output = self.model(images, targets)
        else:
            output = self.model(images)
        return output
