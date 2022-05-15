import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss_orig(preds, target, config):
    bce = nn.BCEWithLogitsLoss()
    ws = config['ws']
    
    loss = 0
    for pred, w in zip(preds['sal'], ws):
        loss += bce(pred, target) * w

    return loss


def Loss(preds, target, config):
    bce = nn.BCEWithLogitsLoss()
    ws = config['ws']
    
    loss = 0
    for pred, w in zip(preds['sal'], ws):
        loss += bce(pred, target) * w
        
        p = torch.sigmoid(pred)
        inter = torch.sum(target * p, dim=(1, 2, 3))
        union = torch.sum(target, dim=(1, 2, 3)) + torch.sum(p, dim=(1, 2, 3)) - inter
        loss += 1 - (inter / union).mean()

    return loss
