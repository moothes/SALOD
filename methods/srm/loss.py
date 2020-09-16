import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(preds, batchs, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    
    slc_loss = 0
    for slc_pred in preds['sal']:
        slc_loss += bce(slc_pred, batchs).mean()
            
    return slc_loss

