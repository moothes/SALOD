import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(pre_dict, target, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    ws = config['ws']
    
    loss = 0
    for pred, w in zip(pre_dict['sal'], ws):
        loss += bce(pred, target).mean() * w

    return loss

