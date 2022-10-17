import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

#from .deeplab_resnet import resnet50_locate
#from .vgg import vgg16_locate

from base.encoder.resnet import resnet50



config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 128}  # no convert layer, no conv6

config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 128}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        #for i in self.bn1.parameters():
        #    i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        #for i in self.bn2.parameters():
        #    i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        #for i in self.bn3.parameters():
        #    i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_locate(nn.Module):
    def __init__(self, block, layers):
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

        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''

    def load_pretrained_model(self, model):
        self.resnet.load_state_dict(model, strict=False)

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

def resnet50_locate():
    model = ResNet_locate(Bottleneck, [3, 4, 6, 3])
    return model


class base_locate(nn.Module):
    def __init__(self, encoder, backbone):
        super(base_locate,self).__init__()
        self.base = encoder
        self.in_planes = 512
        if backbone == 'vgg':
            self.out_planes = [512, 256, 128]
            self.ppms_pre = None
        elif backbone == 'r50':
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
        xs = self.base(x)

        if self.ppms_pre is None:
            xs_1 = xs[-1]
        else:
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
        pools, convs = [],[]
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
            #print(resl.size(), x2.size(), x3.size())
            resl = torch.add(resl, x2)
            resl = F.interpolate(resl, x3.size()[2:], mode='bilinear', align_corners=True)
            resl = self.conv_sum_c(torch.add(resl, x3))
            #resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k ,1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size, mode='bilinear', align_corners=True)
        return x

def extra_layer(base_model_cfg):
    if base_model_cfg == 'vgg':
        config = config_vgg
    elif base_model_cfg == 'r50':
        config = config_resnet
    convert_layers, deep_pool_layers, score_layers = [], [], []
    convert_layers = ConvertLayer(config['convert'])

    for i in range(len(config['deep_pool'][0])):
        deep_pool_layers += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i], config['deep_pool'][2][i], config['deep_pool'][3][i])]

    score_layers = ScoreLayer(config['score'])

    return convert_layers, deep_pool_layers, score_layers


class PoolNet(nn.Module):
    def __init__(self, base_model_cfg, convert_layers, deep_pool_layers, score_layers):
        super(PoolNet, self).__init__()
        self.base_model_cfg = base_model_cfg
        #self.base = base
        self.deep_pool = nn.ModuleList(deep_pool_layers)
        self.score = score_layers
        if self.base_model_cfg == 'r50':
            self.convert = convert_layers

    def forward(self, conv2merge, infos, x_size):
        #x_size = x.size()
        #conv2merge, infos = self.base(x)
        if self.base_model_cfg == 'r50':
            conv2merge = self.convert(conv2merge)
        else:
            conv2merge = conv2merge[1:]
        conv2merge = conv2merge[::-1]

        edge_merge = []
        conv2merge[0] = F.interpolate(conv2merge[0], conv2merge[1].size()[2:], mode='bilinear')
        #print(conv2merge[0].size(), conv2merge[1].size())
        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])

        merge = self.deep_pool[-1](merge)
        merge = self.score(merge, x_size)
        out_dict = {}
        out_dict['sal'] = [merge]
        out_dict['final'] = merge
        return out_dict


class Network(nn.Module):
    #def __init__(self, encoder, backbone):
    def __init__(self, config, encoder, ft):
        super(Network,self).__init__()
        self.encoder = encoder
        self.in_planes = 512
        if config['backbone'] == 'vgg':
            self.out_planes = [512, 256, 128]
            self.ppms_pre = None
        elif config['backbone'] == 'r50':
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
        
        self.decoder = PoolNet(config['backbone'], *extra_layer(config['backbone']))

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        xs = self.encoder(x)

        if self.ppms_pre is None:
            xs_1 = xs[-1]
        else:
            xs_1 = self.ppms_pre(xs[-1])
        xls = [xs_1]
        
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs_1), xs_1.size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))
        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        out_dict = self.decoder(xs, infos, x_size)
        return out_dict


def Network_orig(config, encoder, ft):
    #if config['backbone'] == 'vgg16':
    #    model = PoolNet(config['backbone'], *extra_layer(config['backbone'], vgg16_locate()))
    #elif config['backbone'] == 'resnet':
    #    model = PoolNet(config['backbone'], *extra_layer(config['backbone'], resnet50_locate()))
    temp = base_locate(encoder, config['backbone'])
    model = PoolNet(config['backbone'], *extra_layer(config['backbone'], temp))
    
    #model.apply(weights_init)
    #pre = torch.load('../PretrainModel/resnet50.pth')
    #exist = model.base.resnet.state_dict()
    #for k,v in pre.items():
    #    if k in exist.keys():
    #        exist[k] = v
    #model.base.resnet.load_state_dict(exist)
    
    #model.eval()
    #model.load_state_dict(torch.load('../PretrainModel/resnet50_caffe.pth'))
    #model.base.load_pretrained_model(torch.load('../PretrainModel/resnet50_caffe.pth'))
    #model.apply(weights_init)
    return model

def build_model(base_model_cfg='vgg'):
    if base_model_cfg == 'vgg':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, vgg16_locate()))
    elif base_model_cfg == 'r50':
        return PoolNet(base_model_cfg, *extra_layer(base_model_cfg, resnet50_locate()))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
