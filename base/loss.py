import torch
from torch import nn
from torch.nn import functional as F
from util import *

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

def SSIM(preds, target, config, window_size=11):
    pred = torch.sigmoid(preds['final'])
    
    c = pred.size()[1]
    window = create_window(window_size, c).cuda()
    ssim_loss = 1 - _ssim(pred, target, window, window_size, c)
    return ssim_loss

def BCE(preds, target, config):
    bce = nn.BCEWithLogitsLoss()
    loss = bce(preds['final'], target)
    return loss

def CTLoss(preds, target, config):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    wm = label_edge_prediction(target) * 4 + 1
    loss = (bce(preds['final'], target) * wm).mean()
    return loss

def IOU(preds, target, config):
    pred = torch.sigmoid(preds['final'])

    inter = torch.sum(target * pred, dim=(1, 2, 3))
    union = torch.sum(target, dim=(1, 2, 3)) + torch.sum(pred, dim=(1, 2, 3)) - inter
    iou_loss = 1 - (inter / union).mean()
    return iou_loss

def DICE(preds, target, config):
    pred = torch.sigmoid(preds['final'])

    ab = torch.sum(pred * target, dim=(1, 2, 3))
    a = torch.sum(pred, dim=(1, 2, 3))
    b = torch.sum(target, dim=(1, 2, 3))
    # Dice loss with Laplace smoothing
    dice_loss = 1 - (2 * (ab + 1) / (a + b + 1)).mean()
    return dice_loss
    
def Edge(preds, target, config):
    bce = nn.BCEWithLogitsLoss()
    loss = bce(preds['final'], label_edge_prediction(target))
    return loss


loss_dict = {'b': BCE, 's': SSIM, 'i': IOU, 'd': DICE, 'e': Edge, 'c': CTLoss}

#def Loss_factory(config):
class Loss_factory(nn.Module):
    def __init__(self, config):
        super(Loss_factory, self).__init__()
        if config['lw'] == '':
            lw = [1, ] * len(config['loss'])
        else:
            lw = [float(w) for w in config['lw'].split(',')]
        assert len(config['loss']) == len(lw), 'Length of loss and loss_weight should be equal.'
        
        self.losses = config['loss']
        self.lw = lw
        
    def forward(self, preds, target, config):
        loss = 0
        for loss_name, w in zip(self.losses, self.lw):
            loss += loss_dict[loss_name](preds, target, config) * w
        
        return loss
