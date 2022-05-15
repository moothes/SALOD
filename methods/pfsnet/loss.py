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
    bce = nn.BCEWithLogitsLoss(reduction='none')
    fnl_loss = bce(preds['final'], target.gt(0.5).float()).mean()
    fnl_loss += IOU(preds['final'], target.gt(0.5).float(), config)

    return fnl_loss

