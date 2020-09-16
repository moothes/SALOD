import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable


def Loss(preds, batchs, args):
    bce = nn.BCEWithLogitsLoss()
    
    fnl_loss = bce(preds['final'], batchs)
    for pred in preds['sal']:
        fnl_loss += bce(pred, batchs)
    #fnl_loss = bce(pred['final'], batchs)
    
    return fnl_loss