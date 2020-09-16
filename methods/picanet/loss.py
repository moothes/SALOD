import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(preds, target, config):
    # preds: Dict type. Customarily, preds['final'] is the final prediction without sigmoid.
    ws = config['ws']
    bce = nn.BCEWithLogitsLoss(reduction='none')
    
    fnl_loss = 0
    #print(preds.keys())
    for pred, w in zip(preds['sal'], ws):
        scale = config['size'] // pred.size()[-1]
        tar = F.max_pool2d(target, scale, scale)
        #print(pred.size())
        fnl_loss += bce(pred, tar).mean() * w

    return fnl_loss

