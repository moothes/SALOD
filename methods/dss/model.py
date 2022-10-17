import torch
from torch import nn
from torch.nn import init

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet50

# vgg choice
base = {'dss': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
# extend vgg choice --- follow the paper, you can change it
extra = {'vgg': [(64, 128, 3, [8, 16, 32, 64]), (128, 128, 3, [4, 8, 16, 32]), (256, 256, 5, [8, 16]),
                 (512, 256, 5, [4, 8]), (512, 512, 5, []), (512, 512, 7, [])], 
         'r50': [(64, 128, 3, [8, 16, 32, 64]), (256, 128, 3, [4, 8, 16, 32]), (512, 256, 5, [8, 16]),
                 (1024, 256, 5, [4, 8]), (2048, 512, 5, []), (2048, 512, 7, [])]}
connect = {'dss': [[2, 3, 4, 5], [2, 3, 4, 5], [4, 5], [4, 5], [], []]}


# vgg16
'''
def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers
'''

# feature map before sigmoid: build the connection and deconvolution
class ConcatLayer(nn.Module):
    def __init__(self, list_k, k, scale=True):
        super(ConcatLayer, self).__init__()
        l, up, self.scale = len(list_k), [], scale
        for i in range(l):
            up.append(nn.ConvTranspose2d(1, 1, list_k[i], list_k[i] // 2, list_k[i] // 4))
        self.upconv = nn.ModuleList(up)
        self.conv = nn.Conv2d(l + 1, 1, 1, 1)
        self.deconv = nn.ConvTranspose2d(1, 1, k * 2, k, k // 2) if scale else None

    def forward(self, x, list_x):
        #print(x.size())
        #for x1 in list_x:
        #    print('list: ', x1.size())
        elem_x = [x]
        for i, elem in enumerate(list_x):
            elem_x.append(self.upconv[i](elem))
        #for x1 in elem_x:
        #    
        #    print(x1.size())
        if self.scale:
            out = self.deconv(self.conv(torch.cat(elem_x, dim=1)))
        else:
            out = self.conv(torch.cat(elem_x, dim=1))
        return out


# extend vgg: side outputs
class FeatLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(FeatLayer, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, channel, k, 1, k // 2), nn.ReLU(inplace=True),
                                  nn.Conv2d(channel, 1, 1, 1))

    def forward(self, x):
        return self.main(x)


# fusion features
class FusionLayer(nn.Module):
    def __init__(self, nums=6):
        super(FusionLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(nums))
        self.nums = nums
        self._reset_parameters()

    def _reset_parameters(self):
        init.constant_(self.weights, 1 / self.nums)

    def forward(self, x):
        for i in range(self.nums):
            out = self.weights[i] * x[i] if i == 0 else out + self.weights[i] * x[i]
        return out


# extra part
def extra_layer(vgg, backbone):
    feat_layers, concat_layers, scale = [], [], 1
    scale = 2 if backbone == 'r50' else 1
    cfg = extra[backbone]
    for k, v in enumerate(cfg):
        # side output (paper: figure 3)
        feat_layers += [FeatLayer(v[0], v[1], v[2])]
        # feature map before sigmoid
        concat_layers += [ConcatLayer(v[3], scale, k != 0 or backbone == 'r50')]
        scale *= 2
    return vgg, feat_layers, concat_layers


# DSS network
# Note: if you use other backbone network, please change extract
class DSS(nn.Module):
    def __init__(self, base, feat_layers, concat_layers, connect, extract=[3, 8, 15, 22, 29], v2=True):
        super(DSS, self).__init__()
        self.extract = extract
        self.connect = connect
        self.base = base
        self.feat = nn.ModuleList(feat_layers)
        self.comb = nn.ModuleList(concat_layers)
        self.pool = nn.AvgPool2d(3, 2, 1)
        self.v2 = v2
        if v2: self.fuse = FusionLayer()

    def forward(self, x, phase='test'):
        prob, back, y, num = list(), list(), list(), 0
        xs = self.base(x)
        #for x1 in xs:
        #    print(x1.size())
        for k in range(len(xs)):
            y.append(self.feat[num](xs[k]))
            num += 1
        # side output
        y.append(self.feat[num](self.pool(xs[-1])))
        #print('y: *************')
        for i, k in enumerate(range(len(y))):
            back.append(self.comb[i](y[i], [y[j] for j in self.connect[i]]))
        
        #for x1 in back:
        #    print(x1.size())
        # fusion map
        if self.v2:
            # version2: learning fusion
            back.append(self.fuse(back))
        else:
            # version1: mean fusion
            back.append(torch.cat(back, dim=1).mean(dim=1, keepdim=True))
        
        for yss in y:
            print(yss.shape)
        
        out_dict = {}
        
        out_dict['sal'] = back
        out_dict['final'] = back[-1]
        #out_dict['final'] = torch.cat([back[i] for i in [1, 2, 3, 6]], dim=1).mean(dim=1, keepdim=True)
        #for i in back: prob.append(torch.sigmoid(i))
        return out_dict


# build the whole network
def build_model():
    return DSS(*extra_layer(vgg(pretrained=True), dss), connect['dss'])

def Network(config, encoder, feat):
    '''
    if config['backbone'] == 'vgg':
        model = DSS(*extra_layer(encoder, 'vgg'), connect['dss'])
    elif config['backbone'] == 'resnet':
        model = DSS(*extra_layer(encoder, 'resnet'), connect['dss'])
    '''
    model = DSS(*extra_layer(encoder, config['backbone']), connect['dss'])
    
    return model

# weight init
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = build_model()
    img = torch.randn(1, 3, 64, 64)
    net = net.to(torch.device('cuda:0'))
    img = img.to(torch.device('cuda:0'))
    out = net(img)
    k = [out[x] for x in [1, 2, 3, 6]]
    print(len(k))
    # for param in net.parameters():
    #     print(param)
