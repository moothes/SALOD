import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(preds, target, config):
    # preds: Dict type. Customarily, preds['final'] is the final prediction without sigmoid.
    loss = 0
    for pred in preds['sal']:
        bce = nn.BCEWithLogitsLoss(reduction='none')
        loss += bce(pred, target.gt(0.5).float()).mean()

    return loss

