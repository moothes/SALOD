import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable


def Loss(pred, batchs, args):
    #print(pred['final'].size(), batchs.size())
    #bce = nn.BCEWithLogitsLoss(reduction='sum')
    #print(torch.max(pred['final']), torch.max(batchs))
    fnl_loss = F.binary_cross_entropy_with_logits(pred['final'], batchs, reduction='sum')
    #fnl_loss = bce(pred['final'], batchs)
    
    return fnl_loss
    '''
    loss_map = bce(pred['final'], slc_gt)
    
    pos_loss = (loss_map * slc_gt).sum() / slc_gt.sum()
    neg_loss = (loss_map * neg_mask).sum() / neg_mask.sum()
    

    return pos_loss + neg_loss
    '''





