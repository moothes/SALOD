import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        self.encoder = encoder
        self.decoder = nn.Conv2d(feat[-1], 1, 1)
    
    def forward(self, x, phase='test'):
        # phase: enable different operations in training/testing phases. Useful in several networks.
        x_size = x.size()[2:]
        enc_feats = self.encoder(x)
        x = self.decoder(enc_feats[-1])
        pred = nn.functional.interpolate(x, size=x_size, mode='bilinear')
        
        # All output are save in a Dict, while the key of final prediction is 'final'
        OutDict = {}
        OutDict['final'] = pred
        return OutDict
