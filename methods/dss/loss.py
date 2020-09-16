from torch import nn
import torch.nn.functional as F



def Loss(pred, batchs, args):
    weight=[0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1]
    fnl_loss = 0
    #fnl_loss += F.binary_cross_entropy_with_logits(pred['final'], batchs)
    #print(len(weight), len(pred['sal']))
    #print(pred['final'].size())
    for pre, w in zip(pred['sal'], weight):
        fnl_loss += F.binary_cross_entropy_with_logits(pre, batchs) * w
    
    return fnl_loss