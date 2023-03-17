import torch
import math
from torch import nn
from torch.nn import functional as F


custom_config = {'base'      : {'strategy': 'base_adam',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s



class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False, residual=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        self.residual = residual
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x1 = self.conv(x)
        if self.bn is not None:
            x1 = self.bn(x1)
        if self.residual and x1.shape[1] == x.shape[1]:
            x1 = x + x1
        if self.act is not None:
            x1 = self.act(x1)

        return x1


class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
            bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                               dilation=1, groups=groups, bias=bias,
                               use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x


class ReceptiveConv(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=24, scale=4, dilation=None, aggregation=True, use_dwconv=False):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, self.width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width*scale)
        #self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            if use_dwconv:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], groups=self.width, bias=False))
            else:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, \
                    padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.aggregation = aggregation

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if self.aggregation:
                sp = spx[i] if i == 0 else sp + spx[i]
            else:
                sp = spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, residual=True):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        if self.stride == 1 and inp == oup:
            self.use_res_connect = residual
        else:
            self.use_res_connect = False

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, ksize=1, pad=0))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
                    #arch='mobilenetv2', pretrained=None, use_carafe=True,
                    #enc_channels=[64, 128, 256, 512, 512, 256, 256],
                    #dec_channels=[32, 64, 128, 128, 256, 256, 256], freeze_s1=False):
        super(Network, self).__init__()
        
        arch = config['backbone']
        self.encoder = encoder

        if config['backbone'] == 'vgg':
            feat.append(256)
            feat.append(256)
            enc_channels=feat
            dec_channels=[32, 64, 128, 128, 256, 256, 256]
        elif config['backbone'] == 'resnet50':
            feat.append(1024)
            feat.append(1024)
            enc_channels=feat
            dec_channels=[32, 64, 256, 512, 512, 128, 128]
        elif config['backbone'] == 'mobilenetv2':
            feat.append(40)
            feat.append(40)
            enc_channels=feat
            dec_channels=[16, 24, 32, 40, 40, 40, 40]
        else:
            feat.append(64)
            feat.append(64)
            enc_channels=feat
            dec_channels=[64, 64, 256, 512, 512, 128, 128]
        

        use_dwconv = 'mobile' in config['backbone']
        
        if 'vgg' in arch:
            self.conv6 = nn.Sequential(nn.MaxPool2d(2,2,0),
            ConvBNReLU(enc_channels[-3], enc_channels[-2]),                                   
                                       ConvBNReLU(enc_channels[-2], enc_channels[-2], residual=False),
                                      )
            self.conv7 = nn.Sequential(nn.MaxPool2d(2,2,0),
            ConvBNReLU(enc_channels[-2], enc_channels[-1]),
                                       ConvBNReLU(enc_channels[-1], enc_channels[-1], residual=False),
                                      )
        elif 'resnet50' in arch:
            self.inplanes = enc_channels[-3]
            self.base_width = 64
            self.conv6 = nn.Sequential(
                                       self._make_layer(enc_channels[-2] // 4, 2, stride=2),
                                      )
            self.conv7 = nn.Sequential(
                                       self._make_layer(enc_channels[-2] // 4, 2, stride=2),
                                      )
        elif 'mobilenet' in arch:
            self.conv6 = nn.Sequential(
                                       InvertedResidual(enc_channels[-3], enc_channels[-2], stride=2),
                                       InvertedResidual(enc_channels[-2], enc_channels[-2]),
                                      )
            self.conv7 = nn.Sequential(
                                       InvertedResidual(enc_channels[-2], enc_channels[-1], stride=2),
                                       InvertedResidual(enc_channels[-1], enc_channels[-1]),
                                      )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fpn = CustomDecoder(enc_channels, dec_channels, use_dwconv=use_dwconv)
        
        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)
        
    
    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        groups = 1
        expansion = 4
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample, groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=groups,
                                base_width=self.base_width, dilation=1,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, phase='test'):
        conv1, conv2, conv3, conv4, conv5 = self.encoder(x)
        
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        attention = torch.sigmoid(self.gap(conv7))

        features = self.fpn([conv1, conv2, conv3, conv4, conv5, conv6, conv7], attention)
        
        saliency_maps = []
        for idx, feature in enumerate(features[:5]):
            saliency_maps.append(F.interpolate(
                    getattr(self, 'cls' + str(idx + 1))(feature),
                    x.shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )

        #return torch.sigmoid(torch.cat(saliency_maps, dim=1))
        
        sal_pred = torch.cat(saliency_maps, dim=1)
        
        OutDict = {}
        OutDict['final'] = saliency_maps[0]
        OutDict['sal'] = saliency_maps
        return OutDict



class CustomDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_dwconv=False):
        super(CustomDecoder, self).__init__()
        self.inners_a = nn.ModuleList()
        self.inners_b = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))
        
        self.fuse = nn.ModuleList()
        dilation = [[1, 2, 4, 8]] * (len(in_channels) - 4) + [[1, 2, 3, 4]] * 2 + [[1, 1, 1, 1]] * 2
        baseWidth = [32] * (len(in_channels) - 5) + [24] * 5
        #print("using dwconv:", use_dwconv)
        for i in range(len(in_channels)):
            self.fuse.append(nn.Sequential(
                    ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=use_dwconv),
                    ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i], use_dwconv=use_dwconv)))

    def forward(self, features, att=None):
        if att is not None:
            stage_result = self.fuse[-1](self.inners_a[-1](features[-1] * att))
        else:
            stage_result = self.fuse[-1](self.inners_a[-1](features[-1]))
        results = [stage_result]
        num_mul_att = 1
        for idx in range(len(features) - 2, -1, -1):
            inner_top_down = F.interpolate(self.inners_b[idx](stage_result),
                                           size=features[idx].shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            if att is not None and att.shape[1] == features[idx].shape[1] and num_mul_att:
                features[idx] = features[idx] * att
                num_mul_att -= 1
            inner_lateral = self.inners_a[idx](features[idx])
            stage_result = self.fuse[idx](torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)

        return results
