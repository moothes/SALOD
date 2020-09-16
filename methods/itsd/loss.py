import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(preds, batchs, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    slc_gt = batchs.cuda()
    ws = config['ws']

    slc_loss, ctr_loss = 0, 0
    
    for slc_pred, ctr_pred, l in zip(preds['sal'], preds['edge'], ws):
        scale = int(slc_gt.size(-1) / slc_pred.size(-1))
        ys = F.avg_pool2d(slc_gt, kernel_size=scale, stride=scale).gt(0.5).float()
        yc = label_edge_prediction(ys)
        #yc = F.max_pool2d(yc, 3, 1, 1)
        
        pc = ctr_pred.sigmoid_().float()
        w = torch.where(pc > yc, pc, yc)

        slc_loss += (bce(slc_pred, ys) * (w * 4 + 1)).mean() * l
            
        if ctr_pred is not None:
            ctr_loss += bce(ctr_pred, yc).mean() * l
    
    ctr_gt = label_edge_prediction(slc_gt)
    pc = F.interpolate(pc, size=slc_gt.size()[-2:], mode='bilinear').squeeze(1).float()
    w = torch.where(pc > ctr_gt, pc, ctr_gt)
    fnl_loss = (bce(preds['final'], slc_gt.gt(0.5).float()) * (w * 4 + 1)).mean() * ws[-1]

    return fnl_loss + ctr_loss + slc_loss

