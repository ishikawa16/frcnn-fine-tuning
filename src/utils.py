import json
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


def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        for k, v in config_args.items():
            setattr(args, k, v)
    del args.config
    return args


def make_iou_list(gt_datum, pred_datum):
    ret =[]
    gt_labels = gt_datum["labels"].to("cpu").detach().numpy().copy()
    gt_boxes = gt_datum["boxes"].to("cpu").detach().numpy().copy()
    for label, gt_bb in zip(gt_labels, gt_boxes):
        pred_bbs = pred_datum["boxes"][pred_datum["labels"] == label].to("cpu").detach().numpy().copy()
        iou, pred_bb = compute_iou(gt_bb, pred_bbs)
        ret.append({"gt_bb": gt_bb, "pred_bb": pred_bb, "label": label, "iou": iou})

    return ret


def compute_iou(gt_bb, pred_bbs):
    gt_area = (gt_bb[2] - gt_bb[0]) * (gt_bb[3] - gt_bb[1])
    pred_areas = (pred_bbs[:, 2] - pred_bbs[:, 0]) * (pred_bbs[:, 3] - pred_bbs[:, 1])

    intersect_lx = np.maximum(gt_bb[0], pred_bbs[:, 0])
    intersect_ty = np.maximum(gt_bb[1], pred_bbs[:, 1])
    intersect_rx = np.minimum(gt_bb[2], pred_bbs[:, 2])
    intersect_by = np.minimum(gt_bb[3], pred_bbs[:, 3])
    width = np.maximum(0.0, (intersect_rx - intersect_lx))
    height = np.maximum(0.0, (intersect_by - intersect_ty))
    intersect = width * height

    ious = intersect / (gt_area + pred_areas - intersect)
    iou_max_idx = np.argmax(ious)
    iou_max = ious[iou_max_idx]
    pred_bb_max = pred_bbs[iou_max_idx, :].squeeze()

    return iou_max, pred_bb_max
