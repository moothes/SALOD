import torch
from torch import nn
import torch.nn.functional as F

config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128], [True, True, True, False], [True, True, True, False]], 'score': 128}  # no convert layer, no conv6
config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,256,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 128}

custom_config = {'base'      : {'strategy': 'adam_base',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }

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
    elif base_model_cfg == 'resnet50':
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
        if self.base_model_cfg == 'resnet50':
            self.convert = convert_layers

    def forward(self, conv2merge, infos, x_size):
        #x_size = x.size()
        #conv2merge, infos = self.base(x)
        if self.base_model_cfg == 'resnet50':
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
        elif config['backbone'] == 'resnet50':
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
