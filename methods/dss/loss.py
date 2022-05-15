from torch import nn
import torch.nn.functional as F
from util import *


def CTLoss(preds, target, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    wm = F.avg_pool2d(label_edge_prediction(target), 3, stride=1, padding=1) * 4 + 1
    loss = (bce(preds, target) * wm).mean()
    return loss
    
    
def Fscore(preds, target, config):
    wm = F.avg_pool2d(label_edge_prediction(target), 3, stride=1, padding=1) * 0.8 + 0.2
    pred = torch.sigmoid(preds)
    tp = wm * pred * target
    pred = wm * pred
    target = wm * target
    
    fs = 1.3 * tp.sum(dim=(1, 2, 3)) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) * 0.3)
    loss = 1 - fs.mean()
    
    return loss

def cff_loss(preds, target, config):
    c = CTLoss(preds, target, config)
    f = Fscore(preds, target, config)
    
    return c + 2 * f


def Loss_new(pred, target, config):
    weight=[1, 1, 1, 1, 1, 1, 1]
    fnl_loss = 0
    for pre, w in zip(pred['sal'], weight):
        fnl_loss += cff_loss(pre, target, config) * w
    
    return fnl_loss

def Loss(pred, batchs, args):
    weight=[1, 1, 1, 1, 1, 1, 1]
    fnl_loss = 0
    for pre, w in zip(pred['sal'], weight):
        fnl_loss += F.binary_cross_entropy_with_logits(pre, batchs) * w
    
    return fnl_loss