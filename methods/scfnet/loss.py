import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(preds, target, config):
    # preds: Dict type. Customarily, preds['final'] is the final prediction without sigmoid.
    bce = nn.BCEWithLogitsLoss(reduction='none')
    fnl_loss = bce(preds['final'], target.gt(0.5).float()).mean()

    return fnl_loss

