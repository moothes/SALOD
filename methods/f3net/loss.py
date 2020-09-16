import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *

def Loss(preds, mask, config):
    ws = config['ws']
    loss = 0
    for pred, w in zip(preds['sal'], ws):
        weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
        #print(pred.size(), mask.size())
        wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

        pred  = torch.sigmoid(pred)
        inter = ((pred*mask)*weit).sum(dim=(2,3))
        union = ((pred+mask)*weit).sum(dim=(2,3))
        wiou  = 1-(inter+1)/(union-inter+1)
        loss += (wbce+wiou).mean() * w
    return loss
'''
def Loss(X, batchs, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    slc_gt = batchs.cuda()
    ws = config['ws']
    #ctr_gt = torch.tensor(batchs['C']).cuda().float()

    slc_loss, ctr_loss = 0, 0
    
    for slc_pred, ctr_pred, l in zip(X['slc'], X['ctr'], ws):
        scale = int(slc_gt.size(-1) / slc_pred.size(-1))
        ys = F.avg_pool2d(slc_gt, kernel_size=scale, stride=scale).gt(0.5).float()
        yc = label_edge_prediction(ys) #F.max_pool2d(ctr_gt, kernel_size=scale, stride=scale).float()
        
        #slc_p = slc_pred.unsqueeze(1)
        
        # contour loss
        #w = torch.yc

        # ACT loss
        pc = ctr_pred.sigmoid_().float()
        w = torch.where(pc > yc, pc, yc)

        #print(slc_pred.size(), ys.size())
        slc_loss += (bce(slc_pred, ys) * (w * 4 + 1)).mean() * l
            
        if ctr_pred is not None:
            #ctr_pred = ctr_pred.unsqueeze(1)
            ctr_loss += bce(ctr_pred, yc).mean() * l
    
    ctr_gt = label_edge_prediction(slc_gt)
    pc = F.interpolate(pc, size=slc_gt.size()[-2:], mode='bilinear').squeeze(1).float()
    w = torch.where(pc > ctr_gt, pc, ctr_gt)
    #print(X['final'].size(), slc_gt.size())
    fnl_loss = (bce(X['final'], slc_gt.gt(0.5).float()) * (w * 4 + 1)).mean() * ws[-1]
    #fnl_loss = (bce(X['final'], slc_gt.gt(0.5).float())).mean()

    return fnl_loss + ctr_loss + slc_loss

'''