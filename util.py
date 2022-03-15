import os
import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import importlib
import scipy
import scipy.ndimage
from torch.optim import SGD, Adam
import torch


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def freeze_bn(model):
    for m in model.base.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def label_edge_prediction_old(label):
    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
    fx = np.reshape(fx, (1, 1, 3, 3))
    fy = np.reshape(fy, (1, 1, 3, 3))
    fx = Variable(torch.from_numpy(fx)).cuda()
    fy = Variable(torch.from_numpy(fy)).cuda()
    contour_th = 1.5
    
    # convert label to edge
    label = label.float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad

def label_edge_prediction(label):
    ero = 1 - F.max_pool2d(1 - label, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(label, kernel_size=5, stride=1, padding=2)            # dilation
    
    edge = dil - ero
    return edge

def mask_normalize(mask):
    return mask/(np.amax(mask)+1e-8)
