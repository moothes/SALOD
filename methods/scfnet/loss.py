import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


# dice loss
def dice_loss(pred, mask):
    pred          = F.sigmoid(pred)
    intersection  = (pred * mask).sum(axis=(2, 3))
    unior         = (pred + mask).sum(axis=(2, 3))
    dice          = (2 * intersection + 1) / (unior + 1)
    dice          = torch.mean(1 - dice)
    return dice


# boundary dice loss and PBSM
def boundary_dice_loss(pred, mask, epoch):
    size = max(1, (13 - ((epoch + 1) // 10) * 2))  # PBSM
    pred = F.sigmoid(pred)
    n    = pred.shape[0]
    mask_boundary = F.max_pool2d(1 - mask, kernel_size=3, stride=1, padding=1)
    mask_boundary -= 1 - mask

    pred_boundary = F.max_pool2d(1 - pred, kernel_size=3, stride=1, padding=1)
    pred_boundary -= 1 - pred
    mask_boundary = F.max_pool2d(mask_boundary, kernel_size=size, stride=1, padding=(size-1)//2)
    pred_boundary = F.max_pool2d(pred_boundary, kernel_size=size, stride=1, padding=(size-1)//2)
    mask_boundary = torch.reshape(mask_boundary, shape=(n, -1))
    pred_boundary = torch.reshape(pred_boundary, shape=(n, -1))

    intersection  = (pred_boundary * mask_boundary).sum(axis=(1))
    unior         = (pred_boundary + mask_boundary).sum(axis=(1))
    dice          = (2 * intersection + 1) / (unior + 1)
    dice          = torch.mean(1 - dice)
    return dice


def Loss(preds, target, config):
    # preds: Dict type. Customarily, preds['final'] is the final prediction without sigmoid.
    ws = [1, 0.5, 0.25, 0.125]
    
    loss = 0
    for pred, w in zip(preds['sal'], ws):
        loss1 = dice_loss(pred, target) + 1 * boundary_dice_loss(pred, target, config['cur_epoch'])
        loss += loss1 * w
        
    return loss

