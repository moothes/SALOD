import torch
import numpy as np
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import label_edge_prediction



def Loss(pred, batchs, args):
    bce = nn.BCEWithLogitsLoss()
    slc_gt = batchs.gt(0.5).float().cuda()
    slc_pred = pred['final']
    fnl_loss = bce(slc_pred, slc_gt)
    
    ctr_gt = label_edge_prediction(slc_gt)
    ctr_pred = pred['edge']
    ctr_loss = bce(ctr_pred, ctr_gt)
    #print(fnl_loss)

    return fnl_loss + ctr_loss





