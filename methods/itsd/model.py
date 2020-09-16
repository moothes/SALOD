import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet


NUM = [3, 2, 2, 1, 1]

def initModule(modules):
    for module in modules:
        if type(module) is nn.Conv2d or type(module) is nn.Linear:
            nn.init.kaiming_normal_(module.weight)

def gen_convs(In, Out, num=1):
    for i in range(num):
        yield nn.Conv2d(In, In, 3, padding=1)
        yield nn.BatchNorm2d(In)
        yield nn.ReLU(inplace=True)

def gen_fuse(In, Out):
    yield nn.Conv2d(In, Out, 3, padding=1)
    yield nn.GroupNorm(Out//2, Out)
    #yield nn.BatchNorm2d(Out)
    yield nn.ReLU(inplace=True)

def cp(x, n=2):
    batch, cat, w, h = x.size()
    xn = x.view(batch, cat//n, n, w, h)
    xn = torch.max(xn, dim=2)[0]
    return xn

def gen_final(In, Out):
    yield nn.Conv2d(In, Out, 3, padding=1)
    yield nn.ReLU(inplace=True)

def decode_conv(layer, c):
    for i in range(4 - layer):
        yield nn.Conv2d(c, c, 3, padding=1)
        yield nn.ReLU(inplace=True)
        yield nn.Upsample(scale_factor=2, mode='bilinear' if i == 2 else 'nearest')

    yield nn.Conv2d(c, 8, 3, padding=1)
    yield nn.ReLU(inplace=True)

class pred_block(nn.Module):
    def __init__(self, In, Out, up=False):
        super(pred_block, self).__init__()

        self.final_conv = nn.Conv2d(In, Out, 3, padding=1)
        self.pr_conv = nn.Conv2d(Out, 4, 3, padding=1)
        self.up = up

    def forward(self, X):
        a = nn.functional.relu(self.final_conv(X))
        a1 = self.pr_conv(a)
        pred = torch.max(a1, dim=1, keepdim=True)[0]
        if self.up: 
            a = nn.functional.interpolate(a, scale_factor=2, mode='bilinear')
        return [a, pred]

class res_block(nn.Module):
    def __init__(self, cat, layer):
        super(res_block, self).__init__()

        if layer:
            self.conv4 = nn.Sequential(*list(gen_fuse(cat, cat // 2)))

        self.convs = nn.Sequential(*list(gen_convs(cat, cat, NUM[layer])))
        self.conv2 = nn.Sequential(*list(gen_fuse(cat, cat//2)))

        self.final = nn.Sequential(*list(gen_final(cat, cat)))
        self.layer = layer
        self.initialize()

    def forward(self, X, encoder):
        if self.layer:
            X = nn.functional.interpolate(X, scale_factor=2, mode='bilinear')
            c = cp(X)
            d = self.conv4(encoder)
            X = torch.cat([c, d], 1)

        X = self.convs(X)
        a = cp(X)
        b = self.conv2(encoder)
        f = torch.cat([a, b], 1)
        f = self.final(f)
        return f

    def initialize(self):
        initModule(self.convs)
        initModule(self.conv2)
        initModule(self.final)

        if self.layer:
            initModule(self.conv4)

class ctr_block(nn.Module):
    def __init__(self, cat, layer):
        super(ctr_block, self).__init__()
        self.conv1 = nn.Sequential(*list(gen_convs(cat, cat, NUM[layer])))
        self.conv2 = nn.Sequential(*list(gen_fuse(cat, cat)))
        self.final = nn.Sequential(*list(gen_final(cat, cat)))
        self.layer = layer
        self.initialize()

    def forward(self, X):
        X = self.conv1(X)
        if self.layer:
            X = nn.functional.interpolate(X, scale_factor=2, mode='bilinear')
        X = self.conv2(X)
        x = self.final(X)
        return x

    def initialize(self):
        initModule(self.conv1)
        initModule(self.conv2)
        initModule(self.final)

class final_block(nn.Module):
    def __init__(self, backbone, channel):
        super(final_block, self).__init__()
        self.slc_decode = nn.ModuleList([nn.Sequential(*list(decode_conv(i, channel))) for i in range(5)])
        self.conv = nn.Conv2d(40, 8, 3, padding=1)
        self.backbone = backbone

    def forward(self, xs, phase):
        feats = [self.slc_decode[i](xs[i]) for i in range(5)]
        x = torch.cat(feats, 1)
        
        x = self.conv(x)
        if not self.backbone.startswith('vgg'):
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear')
        
        scale = 2 if phase == 'test' else 1
        x = torch.max(x, dim=1, keepdim=True)[0] * scale
        return x

class baseU(nn.Module):
    def __init__(self, feat, backbone=False, channel=64):
        super(baseU, self).__init__()
        self.name = 'baseU'
        self.layer = 5

        self.adapters = nn.ModuleList([adapter(in1, channel) for in1 in feat])

        self.slc_blocks = nn.ModuleList([res_block(channel, i) for i in range(self.layer)])
        self.slc_preds = nn.ModuleList([pred_block(channel, channel//2)  for i in range(self.layer)])

        self.ctr_blocks = nn.ModuleList([ctr_block(channel, i) for i in range(self.layer)])
        self.ctr_preds = nn.ModuleList([pred_block(channel, channel//2, up=True)  for i in range(self.layer)])

        self.final = final_block(backbone, channel)

    def forward(self, encoders, phase='test'):
        encoders = [self.adapters[i](e_feat) for i, e_feat in enumerate(encoders)]
        
        slcs, slc_maps = [encoders[-1]], []
        ctrs, ctr_maps = [], []
        stc, cts = None, None

        for i in range(self.layer):
            slc = self.slc_blocks[i](slcs[-1], encoders[self.layer - 1 - i])
            if cts is not None:
                slc = torch.cat([cp(slc), cts], dim=1)
            else:
                ctrs.append(slc)
            stc, slc_map = self.slc_preds[i](slc)

            ctr = self.ctr_blocks[i](ctrs[-1])
            ctr = torch.cat([cp(ctr), stc], dim=1)
            cts, ctr_map = self.ctr_preds[i](ctr)

            slcs.append(slc)
            ctrs.append(ctr)
            slc_maps.append(slc_map)
            ctr_maps.append(ctr_map)

        final = self.final(slcs[1:], phase)

        OutPuts = {'final':final, 'sal':slc_maps, 'edge':ctr_maps}
        return OutPuts

class adapter(nn.Module):
    def __init__(self, in1=64, out=64):
        super(adapter, self).__init__()
        self.reduce = in1 // 64
        self.conv = nn.Conv2d(out, out, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        batch, cat, height, width = X.size()
        X = torch.max(X.view(batch, 64, self.reduce, height, width), dim=2)[0]
        X = self.relu(self.conv(X))

        return X

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        c = 64
        self.encoder = encoder
        self.decoder = baseU(feat, config['backbone'], c)
    
    def forward(self, x, phase='test'):
        enc_feats = self.encoder(x)
        OutDict = self.decoder(enc_feats, phase='test')
        return OutDict
