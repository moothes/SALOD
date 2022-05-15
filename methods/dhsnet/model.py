import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet


import torch
import torch.nn as nn
import torchvision


def gen_conv(In, Out):
    yield nn.Conv2d(In, Out, 3, padding=1)
    yield nn.BatchNorm2d(Out)
    yield nn.ReLU(inplace=True)

class RCL(nn.Module):
    """RCL block in DHS net for salient object detection"""

    def __init__(self, channels):
        """Init module.

        :param channels: number of input channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, 1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Sequential(*list(gen_conv(65, 64)))
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)
        self.convs = nn.ModuleList([nn.Sequential(*list(gen_conv(64, 64))) for i in range(3)])
        
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                for i in m.parameters():
                    i.requires_grad = False
        '''

    def forward(self, x, smr, sig=True):
        out1 = self.conv1(x)
        out1 = self.sigmoid(out1)
        out2 = self.sigmoid(smr)
        out = torch.cat((out1, out2), 1)
        out = self.conv2(out)
        out_share = out
        for i in range(3):
            out = self.convs[i](out)
            out = torch.add(out, out_share)
        out = self.conv4(out)
        return out



class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.encoder = encoder
        self.backbone = config['backbone']
        
        self.atten = nn.Conv2d(feat[-1], 1, 3, padding=1)
        self.layer1 = RCL(feat[-2])  # RCL module 1
        self.layer2 = RCL(feat[-3])  # RCL module 2
        self.layer3 = RCL(feat[-4])  # RCL module 3
        self.layer4 = RCL(feat[-5])  # RCL module 4

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        c1, c2, c3, c4, x = self.encoder(x)
        
        x1 = self.atten(x)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x2 = self.layer1(c4, x1)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = self.layer2(c3, x2)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x4 = self.layer3(c2, x3)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear')
        x5 = self.layer4(c1, x4)
        x5 = F.interpolate(x5, size=x_size, mode='bilinear')
        
        out_dict = {}
        out_dict['sal'] = [x5] #[x1, x2, x3, x4, x5]
        out_dict['final'] = x5
        return out_dict


