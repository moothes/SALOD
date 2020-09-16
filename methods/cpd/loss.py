import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *

def Loss(X, target, config):
    bce = nn.BCEWithLogitsLoss()

    atten_loss = bce(X['sal'], target)
    slc_loss = bce(X['final'], target)
            

    return atten_loss + slc_loss

