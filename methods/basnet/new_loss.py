import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from util import *

import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def _logssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim_map = (ssim_map - torch.min(ssim_map))/(torch.max(ssim_map)-torch.min(ssim_map))
    ssim_map = -torch.log(ssim_map + 1e-8)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class LOGSSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(LOGSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _logssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def IOU(pred, target):
    inter = torch.sum(target * pred, dim=(1, 2, 3))
    union = torch.sum(target, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3)) - inter
    iou_loss = 1 - (inter / union).mean()
    return iou_loss

def bce_ssim_loss(pred,target):

	bce_out = nn.BCELoss(size_average=True)(pred,target)
	ssim_out = 1 - SSIM(window_size=11,size_average=True)(pred,target)
	iou_out = IOU(pred, target)

	loss = bce_out + ssim_out + iou_out

	return loss

def Loss(preds, target, config):
    #loss = bce_ssim_loss(torch.sigmoid(preds['final']), target)
    loss = 0
    for pred in preds['sal']:
        loss += bce_ssim_loss(torch.sigmoid(pred), target)
    return loss

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

def Loss_new(preds, target, config):
    loss = 0
    for pred in preds['sal']:
        loss += cff_loss(pred, target, config)
    return loss
    