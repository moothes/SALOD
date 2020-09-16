import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

from ..base.encoder.vgg import vgg
from ..base.encoder.resnet import resnet50

from util import *


#from .deeplab_resnet import resnet50_locate
#from .vgg import vgg16_locate


config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 128}


class vgg16_locate(nn.Module):
    def __init__(self):
        super(vgg16_locate,self).__init__()
        self.vgg16 = vgg('vgg16', multi=True, pretrain=True)
        self.in_planes = 512
        self.out_planes = [512, 256, 128]

        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrained_model(self, model):
        self.vgg16.load_pretrained_model(model)

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.vgg16(x)

        xls = [xs[-1]]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs[-1]), xs[-1].size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))
        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, infos

        
class ResNet_locate(nn.Module):
    def __init__(self):
        super(ResNet_locate,self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.in_planes = 512
        self.out_planes = [512, 256, 256, 128]

        self.ppms_pre = nn.Conv2d(2048, self.in_planes, 1, 1, bias=False)
        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.resnet(x)

        xs_1 = self.ppms_pre(xs[-1])
        xls = [xs_1]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))

        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, infos

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl

class DeepPoolLayer(nn.Module):
    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [], []
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

def extra_layer(base_model_cfg, vgg):
    if base_model_cfg == 'vgg16':
        config = config_vgg
    elif base_model_cfg == 'resnet':
        config = config_resnet
    convert_layers, deep_pool_layers, score_layers = [], [], []
    convert_layers = ConvertLayer(config['convert'])

    for i in range(len(config['deep_pool'][0])):
        deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]

    score_layers = ScoreLayer(config['score'])

    return vgg, convert_layers, deep_pool_layers, score_layers


class deocder(nn.Module):
    def __init__(self, base_model_cfg, convert_layers, deep_pool_layers, score_layers):
        super(deocder, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers
        if self.base_model_cfg == 'resnet':
            self.convert = convert_layers

    def forward(self, conv2merge, infos, x_size):
        #x_size = x.size()
        #conv2merge, infos = self.encoder(x)
        if self.base_model_cfg == 'resnet':
            conv2merge = self.convert(conv2merge)
        conv2merge = conv2merge[::-1]

        edge_merge = []
        #for l in conv2merge:
        #    print(l.size())
        #print(conv2merge[0].size())
        conv2merge[0] = F.interpolate(conv2merge[0], conv2merge[1].size()[2:], mode='bilinear')
        #print(conv2merge[0].size())
        #print(conv2merge[0].size(), conv2merge[1].size(), infos[0].size())
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])

        merge = self.deep_pool[-1](merge)
        merge = self.score(merge, x_size)
        
        out_dict = {}
        out_dict['final'] = merge
        return out_dict
    
class PoolNet(nn.Module):
    def __init__(self, base_model_cfg, base, convert_layers, deep_pool_layers, score_layers):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.encoder = base
        self.decoder = deocder(self.base_model_cfg, convert_layers, deep_pool_layers, score_layers)
        #self.decoder.apply(weights_init)

    def forward(self, x):
        x_size = x.size()
        conv2merge, infos = self.encoder(x)
        out = self.decoder(conv2merge, infos, x_size)
        return out
            
def initModule(modules):
    for module in modules:
        if type(module) is nn.Conv2d or type(module) is nn.Linear:
            nn.init.kaiming_normal_(module.weight)

def Network(config):
    if config['backbone'] == 'vgg16':
        model = PoolNet(config['backbone'], *extra_layer(config['backbone'], vgg16_locate()))
    elif config['backbone'] == 'resnet':
        model = PoolNet(config['backbone'], *extra_layer(config['backbone'], ResNet_locate()))
    
    model.apply(weights_init)
    pre_st = torch.load('../PretrainModel/resnet50.pth')
    ex_st = model.encoder.resnet.state_dict()
    for k, v in pre_st.items():
        if k in ex_st.keys():
            #print(k)
            ex_st[k] = v
    #ex_st.update(state_dict)
    model.encoder.resnet.load_state_dict(ex_st)
    model.train()
    #model.eval()
    #model.apply(weights_init)
    return model

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #print(m)
        #nn.init.kaiming_normal_(m.weight)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    
    
'''
class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        backbone = config['backbone']
        self.model = build_model(backbone)
    
    def forward(self, x):
        return self.model(x)
'''