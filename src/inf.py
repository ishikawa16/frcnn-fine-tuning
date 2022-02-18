from PIL import Image

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision import transforms

from roi_heads import RoIHeads
from utils import SaveFeatures, collate_fn


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2

model = fasterrcnn_resnet50_fpn(pretrained=True)
box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
box_head = TwoMLPHead(model.backbone.out_channels * box_roi_pool.output_size[0] ** 2, 1024)
box_predictor = FastRCNNPredictor(1024, num_classes)
model.roi_heads = RoIHeads(
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
model.load_state_dict(torch.load('model/alfred_model_e10.pth'))

save_features = SaveFeatures()
model.roi_heads.box_head.fc6.register_forward_hook(save_features)
model.to(device)

model.eval()

image = Image.open('data/alfred_pick_only/images/000001.jpg')
convert_tensor = transforms.ToTensor()
image = convert_tensor(image)
images = [image.to(device)]
outputs = model(images)
