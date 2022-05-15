import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def IOU(preds, target, config):
    pred = torch.sigmoid(preds)

    inter = torch.sum(target * pred, dim=(1, 2, 3))
    union = torch.sum(target, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3)) - inter
    iou_loss = 1 - (inter / union).mean()
    return iou_loss

def Loss(preds, target, config):
    # preds: Dict type. Customarily, preds['final'] is the final prediction without sigmoid.
    loss = 0
    for pred in preds['sal']:
        bce = nn.BCEWithLogitsLoss(reduction='none')
        loss += 0.6 * bce(pred, target.gt(0.5).float()).mean()
        loss += IOU(pred, target, config)
        
    gt_edge = label_edge_prediction(target)
    for pred in preds['edge']:
        bce = nn.BCEWithLogitsLoss(reduction='none')
        loss += bce(pred, gt_edge.gt(0.5).float()).mean()
    return loss

