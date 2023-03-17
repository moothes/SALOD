import torch
from torch import nn
import torch.nn.functional as F


custom_config = {'base'      : {'strategy': 'base_adam',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }
         

class ConvConstract(nn.Module):
    def __init__(self, in_channel):
        super(ConvConstract, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=3, padding=1)
        self.cons1 = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x2 = self.cons1(x)
        return x, x - x2

class Network(nn.Module):
    # resnet based encoder decoder
    #def __init__(self, base, feat_layers, pool_layers):
    def __init__(self, config, encoder, fl):
        super(Network, self).__init__()
        
        self.config = config
        feat_layers = []
        pool_layers = []
        for k, feat in enumerate(fl):
            feat_layers.append(ConvConstract(feat))
            if k == 0:
                pool_layers += [nn.Conv2d(128 * (6 - k), 128 * (5 - k), 1)]
            else:
                pool_layers += [nn.ConvTranspose2d(128 * (6 - k), 128 * (5 - k), 3, 2, 1, 1)]
        
        self.pos = [4, 9, 16, 23, 30]
        self.encoder = encoder
        
        self.feat = nn.ModuleList(feat_layers)
        self.pool = nn.ModuleList(pool_layers)
        self.glob = nn.Sequential(nn.Conv2d(fl[-1], 128, 3), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3),
                                  nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        self.conv_g = nn.Conv2d(128, 1, 1)
        self.conv_l = nn.Conv2d(640, 1, 1)

    def forward(self, x, phase='test'):
        if self.config['backbone'] == 'vgg':
            sources, num = list(), 0
            for k in range(len(self.encoder.features)):
                x = self.encoder.features[k](x)
                if k in self.pos:
                    sources.append(self.feat[num](x))
                    num = num + 1
        elif self.config['backbone'] == 'resnet50':
            ens = self.encoder(x)
            sources = [b(f) for f, b in zip(ens, self.feat)]
            x = ens[-1]
            
        
        #ens = self.encoder(x)
        #sources = [b(f) for f, b in zip(ens, self.feat)]
        for k in range(4, -1, -1):
            if k == 4:
                out = F.relu(self.pool[k](torch.cat([sources[k][0], sources[k][1]], dim=1)), inplace=True)
            else:
                out = self.pool[k](torch.cat([sources[k][0], sources[k][1], out], dim=1)) if k == 0 else F.relu(
                    self.pool[k](torch.cat([sources[k][0], sources[k][1], out], dim=1)), inplace=True)

        a = self.glob(x)
        #print(a.size(), out.size())
        score = self.conv_g(a) + self.conv_l(out)
        score = nn.functional.interpolate(score, scale_factor=2, mode='bilinear')
        out_dict = {}
        out_dict['sal'] = [score]
        out_dict['final'] = score
        #prob = torch.sigmoid(score)
        return out_dict



