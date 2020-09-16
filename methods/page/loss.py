import torch
from torch.nn import functional as F
from torch.autograd import Variable
from util import *


def Loss(X, target, config):
    '''
    bce = nn.BCEWithLogitsLoss()
    edge_gt = label_edge_prediction(target)

    
    slc_loss, ctr_loss = 0, 0
    for sal_pred in X['sal']:
        #print(sal_pred.size(), target.size())
        slc_loss += bce(sal_pred, target)
        
    for edge_pred in X['edge']:
        ctr_loss += bce(edge_pred, edge_gt)
        
    return ctr_loss + slc_loss
    '''
    
    #print(torch.max(X['final']))
    return F.binary_cross_entropy_with_logits(X['final'], target)