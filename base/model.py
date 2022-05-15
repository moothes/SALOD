import importlib
from torch import nn

from torch.nn import functional as F
from .encoder.vgg import vgg
from .encoder.resnet import resnet
from .encoder.efficientnet import EfficientNet
from .encoder.mobile import mobilenet
from .encoder.ghost import ghostnet
from .encoder.res2net import res2net50_14w_8s as res2net
from .encoder.mobilev3 import mobilenetv3


def Network(net_name, config):
    if config['backbone'] == 'vgg':
        encoder = vgg(pretrained=True)
        fl = [64, 128, 256, 512, 512]
    elif config['backbone'] == 'resnet':
        encoder = resnet(pretrained=True)
        fl = [64, 256, 512, 1024, 2048]
    elif config['backbone'] == 'eff':
        encoder = EfficientNet.from_pretrained('efficientnet-b0', weights_path='../PretrainModel/efficientnet-b0-355c32eb.pth')
        fl = [16, 24, 40, 112, 1280]
    elif config['backbone'] == 'mobile':
        encoder = mobilenet()
        fl = [16, 24, 32, 64, 160]
    elif config['backbone'] == 'mobilev3':
        encoder = mobilenetv3(pretrained=True)
        fl = [16, 16, 24, 48, 576]
    elif config['backbone'] == 'ghost':
        encoder = ghostnet()
        fl = [16, 24, 40, 112, 960]
    elif config['backbone'] == 'res2net':
        encoder = res2net(pretrained=True)
        fl = [64, 256, 512, 1024, 2048]
    else:
        encoder = resnet(pretrained=True)
        fl = [64, 256, 512, 1024, 2048]
    
    model = importlib.import_module('methods.{}.model'.format(net_name)).Network(config, encoder, fl)
    config = config
    return model
