import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(preds, batchs, config):
    bce = nn.BCEWithLogitsLoss()
    #bce = nn.BCELoss()
    '''
    slc_loss = 0
    for slc_pred, w in zip(preds['sal'], config['ws']):
        ys = F.interpolate(batchs, size=slc_pred.size()[2:], mode='bilinear')
        #print(torch.max(slc_pred))
        slc_loss += bce(slc_pred, ys) * w
            
    '''
    slc_loss = bce(preds['final'], batchs)
    return slc_loss

