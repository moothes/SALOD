import importlib
from torch import nn

from torch.nn import functional as F
from .encoder.vgg import vgg
from .encoder.resnet import resnet


class Network(nn.Module):
    def __init__(self, net_name, config):
        super(Network, self ).__init__()
        
        if config['backbone'] == 'vgg':
            encoder = vgg(pretrained=True)
            fl = [64, 128, 256, 512, 512]
        elif config['backbone'] == 'resnet':
            encoder = resnet(pretrained=True)
            fl = [64, 256, 512, 1024, 2048]
        else:
            encoder = resnet(pretrained=True)
            fl = [64, 256, 512, 1024, 2048]
        
        self.model = importlib.import_module('methods.{}.model'.format(net_name)).Network(config, encoder, fl)
        self.config = config
    
    def forward(self, x, phase='test'):
        x_size = x.size()
        res = self.model(x, phase='test')
        res['final'] = F.interpolate(res['final'], x_size[2:], mode='bilinear', align_corners=True)
        return res