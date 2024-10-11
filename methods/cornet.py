import torch
from torch import nn
from torch.nn import functional as F

custom_config = {'base'      : {'strategy': 'sgd_f3net',
                                'batch': 8,
                                'loss': 'bce,iou',
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }
         

def up_conv(cin, cout, up=True):
    #yield nn.ConvTranspose2d(cin, cout * 2, 5, stride=2, padding=2)
    #yield nn.ConvTranspose2d(cout * 2, cout, 5, stride=2, padding=2)
    yield nn.Conv2d(cin, cout, 3, padding=1)
    yield nn.GroupNorm(cout//2, cout)
    yield nn.ReLU(inplace=True)
    if up:
        yield nn.Upsample(scale_factor=2, mode='bilinear')

def local_conv(cin, cout):
    yield nn.Conv2d(cin, cout * 2, 3, padding=1)
    #yield nn.BatchNorm2d(cout * 2)
    yield nn.GroupNorm(cout, cout * 2)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')
    yield nn.Conv2d(cout * 2, cout, 3, padding=1)
    #yield nn.BatchNorm2d(cout)
    yield nn.GroupNorm(cout//2, cout)
    yield nn.ReLU(inplace=True)
    yield nn.Upsample(scale_factor=2, mode='bilinear')

class SE_block(nn.Module):
    def __init__(self, feat):
        super(SE_block, self).__init__()
        self.conv = nn.Conv2d(feat, feat, 1)
        #self.gn = nn.GroupNorm(feat // 2, feat)

    def forward(self, x):
        glob_x = F.adaptive_avg_pool2d(x, (1, 1))
        glob_x = torch.sigmoid(self.conv(glob_x))
        x = glob_x * x

        return x

class global_block(nn.Module):
    def __init__(self, config, feat):
        super(global_block, self).__init__()
        self.gconv = nn.Sequential(*list(up_conv(feat[-1], feat[0], False)))
        
        self.se = SE_block(feat[0])
        #self.se = ASPP(feat[0])

    def forward(self, xs):
        glob_x = self.gconv(xs[-1])
        glob_x = self.se(glob_x)

        return glob_x

class region_block(nn.Module):
    def __init__(self, config, feat):
        super(region_block, self).__init__()
        self.conv4 = nn.Sequential(*list(up_conv(feat[3], feat[2])))
        self.conv3 = nn.Sequential(*list(up_conv(feat[2] * 2, feat[0], False)))
        self.se = SE_block(feat[0])

        self.fuse = nn.Conv2d(feat[0] * 2, feat[0], 3, padding=1)
        #self.se1 = SE_block(feat[0])

    def forward(self, xs, glob_x):
        reg_x = self.conv4(xs[3])
        reg_x = self.conv3(torch.cat([xs[2], reg_x], dim=1))
        reg_x = self.se(reg_x)

        glob_x = nn.functional.interpolate(glob_x, size=reg_x.size()[2:], mode='bilinear')

        reg_x = torch.cat([reg_x, glob_x], dim=1)
        reg_x = self.fuse(reg_x)
        #reg_x = self.se1(reg_x)

        return reg_x

class local_block(nn.Module):
    def __init__(self, config, feat):
        super(local_block, self).__init__()
        self.conv2 = nn.Sequential(*list(up_conv(feat[1], feat[0])))
        self.conv1 = nn.Sequential(*list(up_conv(feat[0] * 2, feat[0], False)))
        self.se = SE_block(feat[0])

        self.gb_conv = nn.Sequential(*list(local_conv(feat[0], feat[0])))

        self.fuse = nn.Conv2d(feat[0] * 2, feat[0], 3, padding=1)
        #self.se1 = SE_block(feat[0])

    def forward(self, xs, glob_x):
        loc_x = self.conv2(xs[1])
        loc_x = self.conv1(torch.cat([xs[0], loc_x], dim=1))
        loc_x = self.se(loc_x)

        glob_x = self.gb_conv(glob_x)
        glob_x = nn.functional.interpolate(glob_x, size=xs[0].size()[2:], mode='bilinear')

        loc_x = torch.cat((loc_x, glob_x), dim=1)
        loc_x = self.fuse(loc_x)
        #loc_x = self.se1(loc_x)

        return loc_x


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        self.encoder = encoder
        
        #self.feat = feat[0] // 2
        #self.adapters = nn.ModuleList([nn.Sequential(*list(up_conv(in1, self.feat, False))) for in1 in feat])
        #feat = [feat[0] // 2, ] * 5

        self.global_ = global_block(config, feat)
        self.region = region_block(config, feat)
        self.local = local_block(config, feat)

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        xs = self.encoder(x)
        
        #xs = [self.adapters[i](e_feat) for i, e_feat in enumerate(xs)]
        #for i, x in enumerate(xs):
        #    batch, channel, w, h = x.size()
        #    x = x.view(batch, channel // self.feat, self.feat, w, h)
        #    xs[i] = torch.mean(x, dim=1)
        
        glob_x = self.global_(xs)
        reg_x = self.region(xs, glob_x)
        loc_x = self.local(xs, glob_x)

        reg_x = nn.functional.interpolate(reg_x, size=xs[0].size()[2:], mode='bilinear')
        #xp = F.cosine_similarity(loc_x, reg_x)
        #xp = xp.unsqueeze(dim=1)
        #print(xp.size())
        #pred = (nn.functional.interpolate(xp, size=x_size, mode='bilinear') + 1) / 2
        pred = torch.sum(loc_x * reg_x, dim=1, keepdim=True)
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear')

        OutDict = {}
        OutDict['feat'] = [loc_x, reg_x]
        OutDict['sal'] = [pred, ]
        OutDict['final'] = pred

        return OutDict
