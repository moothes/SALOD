import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(preds, slc_gt, args):
    
    up_edge = preds['edge']
    up_sal = preds['sal']
    #up_sal_f = preds['sal_f']
    
    ctr_gt = label_edge_prediction(slc_gt)
    
    
    edge_loss = []
    for ix in up_edge:
        edge_loss.append(bce2d_new(ix, ctr_gt, reduction='mean'))
    edge_loss = sum(edge_loss)
    
    sal_loss1 = []
    #sal_loss2 = []
    for ix in up_sal:
        sal_loss1.append(F.binary_cross_entropy_with_logits(ix, slc_gt, reduction='mean'))

    #for ix in up_sal_f:
    #    sal_loss2.append(F.binary_cross_entropy_with_logits(ix, slc_gt, reduction='sum'))
    sal_loss = sum(sal_loss1)

    return edge_loss + sal_loss



def bce2d_new(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


