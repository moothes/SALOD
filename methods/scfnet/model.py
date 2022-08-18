import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet



def weight_init(module):
    for n, m in module.named_children():
       # print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_value_0(m.bias)

        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            constant_value_1(m.weight)
            if m.bias is not None:
                constant_value_0(m.bias)
        elif isinstance(m, nn.Linear):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_value_0(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AdaptiveMaxPool2d):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        elif isinstance(m, nn.MaxPool2d):
            pass
        else:
            m.init_weight()


class MappingModule(nn.Module):
    def __init__(self, type, out_c):
        super(MappingModule, self).__init__()
        if type == 'M3_0.5':
            nums = [16, 24, 56, 80]
        elif type == 'R34':
            nums = [64, 128, 256, 512]
        elif type == 'ST':
            nums = [128, 256, 512, 1024]
        else:
            nums = [256, 512, 1024, 2048]
        self.cv1 = nn.Sequential(
            nn.Conv2d(nums[0], out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(nums[1], out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(nums[2], out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(nums[3], out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, out2, out3, out4, out5):
        out2 = self.cv1(out2)
        out3 = self.cv2(out3)
        out4 = self.cv3(out4)
        out5 = self.cv4(out5)
        return out2, out3, out4, out5

    def init_weight(self):
        weight_init(self)


class SCFM(nn.Module):
    def __init__(self, in_c):
        super(SCFM, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )

    def forward(self, fl, fh, fd):
        fdh = F.interpolate(fd, size=fh.shape[2:], mode='bilinear')
        fdh = fh * fdh
        fdh = self.cv1(fh + fdh) 

        fhl = F.interpolate(fh, size=fl.shape[2:], mode='bilinear')
        fhl = fl * fhl
        fhl = self.cv2(fl + fhl)

        fdh = F.interpolate(fdh, size=fl.shape[2:], mode='bilinear')
        fll = self.cv3(fhl + fdh)
        return fll

    def init_weight(self):
        weight_init(self)


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()
        
        backbone_type = config['backbone']
        self.in_c = 64
        
        self.encoder = encoder
        
        self.mp = MappingModule(backbone_type, self.in_c)
        self.cv1 = nn.Sequential(nn.Conv2d(self.in_c, self.in_c, 3, 1, 5, dilation=5), nn.BatchNorm2d(self.in_c), nn.ReLU())
        self.de1 = SCFM(self.in_c)
        self.de2 = SCFM(self.in_c)
        self.de3 = SCFM(self.in_c)
        self.linear2 = nn.Conv2d(self.in_c, 1, 3, 1, 1)
        self.linear3 = nn.Conv2d(self.in_c, 1, 3, 1, 1)
        self.linear4 = nn.Conv2d(self.in_c, 1, 3, 1, 1)
        self.linear5 = nn.Conv2d(self.in_c, 1, 3, 1, 1)
        
        
    def forward(self, x, phase='test'):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x2, x3, x4, x5     = self.mp(x2, x3, x4, x5)
        x5                 = self.cv1(x5)
        x4                 = self.de3(x4, x5, x5)
        x3                 = self.de2(x3, x4, x5)
        x2                 = self.de1(x2, x3, x5)
        x2                 = F.interpolate(self.linear2(x2), mode='bilinear', size=x.shape[2:])
        x3                 = F.interpolate(self.linear3(x3), mode='bilinear', size=x.shape[2:])
        x4                 = F.interpolate(self.linear4(x4), mode='bilinear', size=x.shape[2:])
        x5                 = F.interpolate(self.linear5(x5), mode='bilinear', size=x.shape[2:])
        
        # All output are save in a Dict, while the key of final prediction is 'final'
        OutDict = {}
        OutDict['final'] = x2
        OutDict['sal'] = [x2, x3, x4, x5]
        return OutDict


        
