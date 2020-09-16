import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *

def logit(x):
    eps = 1e-7
    x = torch.clamp(x,eps,1-eps)
    x = torch.log(x / (1 - x))
    return x

def cross_entropy(logits,labels):
    return torch.mean((1 - labels) * logits + torch.log(1 + torch.exp(-logits)))

def weighted_cross_entropy(logits,labels,alpha):
    return torch.mean((1 - alpha) * ((1 - labels) * logits + torch.log(1 + torch.exp(-logits))) + (2 * alpha - 1) * labels * torch.log(1 + torch.exp(-logits)))
'''
class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        laplace = torch.FloatTensor([[-1,-1,-1,],[-1,8,-1],[-1,-1,-1]]).view([1,1,3,3])
        self.laplace = nn.Parameter(data=laplace,requires_grad=False).cuda()
        self.config = config
        
    def torchLaplace(self,x):
        edge = F.conv2d(x,self.laplace,padding=1)
        edge = torch.abs(torch.tanh(edge))
        return edge
        
    def forward(self, X, y_true):
        y_pred = nn.Sigmoid()(X['final'])
        y_true_edge = self.torchLaplace(y_true)
        y_pred_edge = self.torchLaplace(y_pred)
        #edge_loss = cross_entropy(y_pred_edge,y_true_edge)
        edge_loss = 0
        saliency_loss = weighted_cross_entropy(y_pred,y_true,alpha=0.528)
        
        return 0.8 * saliency_loss + 0.2 * edge_loss
'''
def get_edge(x, filter):
    edge = F.conv2d(x, filter, padding=1)
    edge = torch.abs(torch.tanh(edge))
    return edge

def Loss(X, y_true, config):
    #laplace = nn.Parameter(data=torch.FloatTensor([[-1,-1,-1,],[-1,8,-1],[-1,-1,-1]]).view([1,1,3,3]),requires_grad=False).cuda()
    
    y_pred = torch.sigmoid(X['final'])
    #y_true_edge = label_edge_prediction(y_true, laplace)
    #y_pred_edge = label_edge_prediction(y_pred, laplace)
    #edge_loss = cross_entropy(y_pred_edge,y_true_edge)
    saliency_loss = weighted_cross_entropy(y_pred,y_true,alpha=0.528)
    
    return 0.8 * saliency_loss #+ 0.2 * edge_loss
