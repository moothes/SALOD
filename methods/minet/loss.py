import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(X, target, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    bce_loss = bce(X['final'], target.gt(0.5).float()).mean()

    pred = X['final'].sigmoid()
    intersection = pred * target
    numerator = (pred - intersection).sum() + (target - intersection).sum()
    denominator = pred.sum() + target.sum()
    cel_loss = numerator / (denominator + 1e-6)
    return bce_loss + cel_loss

