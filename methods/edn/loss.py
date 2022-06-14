import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *

def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    #print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice

def Loss(preds, target, config):
    # preds: Dict type. Customarily, preds['final'] is the final prediction without sigmoid.
    #bce = nn.BCEWithLogitsLoss(reduction='none')
    #fnl_loss = bce(preds['final'], target.gt(0.5).float()).mean()
    
    loss = 0
    for pred in preds['sal']:
        loss += BCEDiceLoss(torch.sigmoid(pred), target)
    
    return loss

