import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def CTLoss(preds, target, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    wm = F.avg_pool2d(label_edge_prediction(target), 3, stride=1, padding=1) * 4 + 1
    loss = (bce(preds, target) * wm).mean()
    return loss
    
    
def Fscore(preds, target, config):
    wm = F.avg_pool2d(label_edge_prediction(target), 3, stride=1, padding=1) * 0.8 + 0.2
    pred = torch.sigmoid(preds)
    tp = wm * pred * target
    pred = wm * pred
    target = wm * target
    
    fs = 1.3 * tp.sum(dim=(1, 2, 3)) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) * 0.3)
    loss = 1 - fs.mean()
    
    return loss

def cff_loss(preds, target, config):
    c = CTLoss(preds, target, config)
    f = Fscore(preds, target, config)
    
    return c + 2 * f

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

        slc_loss += cff_loss(slc_pred, ys, config) #(bce(slc_pred, ys) * (w * 4 + 1)).mean() * l
            
        if ctr_pred is not None:
            ctr_loss += bce(ctr_pred, yc).mean() * l
    
    ctr_gt = label_edge_prediction(slc_gt)
    pc = F.interpolate(pc, size=slc_gt.size()[-2:], mode='bilinear').squeeze(1).float()
    w = torch.where(pc > ctr_gt, pc, ctr_gt)
    fnl_loss = cff_loss(preds['final'], slc_gt.gt(0.5).float(), config)
    #fnl_loss = (bce(preds['final'], slc_gt.gt(0.5).float()) * (w * 4 + 1)).mean() * ws[-1]

    return fnl_loss + ctr_loss + slc_loss



def Loss_orig(preds, target, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    target = target.gt(0.5).float()
    ws = config['ws']

    slc_loss, ctr_loss = 0, 0
    
    for slc_pred, ctr_pred, l in zip(preds['sal'], preds['edge'], ws):
        scale = int(target.size(-1) / slc_pred.size(-1))
        ys = F.avg_pool2d(target, kernel_size=scale, stride=scale).gt(0.5).float()
        yc = label_edge_prediction(ys)
        
        pc = ctr_pred.sigmoid_().float()
        w = torch.where(pc > yc, pc, yc)

        slc_loss += (bce(slc_pred, ys) * (w * 4 + 1)).mean() * l
            
        if ctr_pred is not None:
            ctr_loss += bce(ctr_pred, yc).mean() * l
    
    ctr_gt = label_edge_prediction(target)
    pc = F.interpolate(pc, size=target.size()[-2:], mode='bilinear').squeeze(1).float()
    w = torch.where(pc > ctr_gt, pc, ctr_gt)
    fnl_loss = (bce(preds['final'], target) * (w * 4 + 1)).mean() * ws[-1]

    return fnl_loss + ctr_loss + slc_loss

