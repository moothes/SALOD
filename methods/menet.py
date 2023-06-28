#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


custom_config = {'base'      : {'strategy': 'base_adam',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }

def weight_init(module):
    for n, m in module.named_children():
        #print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Sigmoid):
            pass
        else:
            m.initialize()


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def initialize(self):
        pass


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        #self.resnet = timm.create_model(model_name="resnet101", pretrained=True, in_chans=3, features_only=True)
        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5 = self.resnet(x)
        return out1, out2, out3, out4, out5

    def initialize(self):
        pass


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

    def initialize(self):
        pass


class Enhance_Decoder(nn.Module):
    def __init__(self):
        super(Enhance_Decoder, self).__init__()
        self.conv0 = self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64)
        self.conv1 = self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64)
        self.conv2 = self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64)
        self.conv3 = self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64)
        self.conv4 = self._make_pred_layer(Classifier_Module, [1, 6, 8, 12], [1, 6, 8, 12], 64)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)

        self.mea0 = ME_Attention()
        self.mea1 = ME_Attention()
        self.mea2 = ME_Attention()
        self.mea3 = ME_Attention()
        self.mea4 = ME_Attention()

        self.conv = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1, relu=True)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, input1, input2=[0, 0, 0, 0, 0], input3=[0, 0, 0, 0, 0], input4=[0, 0, 0, 0, 0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0] + input3[0] + input4[0])), inplace=True)
        out0 = out0 * self.mea0(out0)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1] + input2[1] + + input3[1] + input4[1] + out0)), inplace=True)
        out1 = out1 * self.mea1(out1)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2] + input2[2] + + input3[2] + input4[2] + out1)), inplace=True)
        out2 = out2 * self.mea2(out2)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3] + input2[3] + + input3[3] + input4[3] + out2)), inplace=True)
        out3 = out3 * self.mea3(out3)
        out3 = F.interpolate(out3, size=input1[4].size()[2:], mode='bilinear')
        out4 = F.relu(self.bn4(self.conv4(input1[4] + input2[4] + + input3[4] + input4[4] + out3)), inplace=True)
        out4 = out4 * self.mea4(out4)
        out4 = self.conv(out4)

        return out4

    def initialize(self):
        weight_init(self)


class Enhance_Encoder(nn.Module):
    def __init__(self):
        super(Enhance_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b   = nn.BatchNorm2d(64)
        self.conv5b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5b = nn.BatchNorm2d(64)

        self.conv1d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d   = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d   = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d   = nn.BatchNorm2d(64)
        self.conv4d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d   = nn.BatchNorm2d(64)
        self.conv5d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5d = nn.BatchNorm2d(64)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)
        out5 = F.max_pool2d(out4, kernel_size=2, stride=2)
        out5 = F.relu(self.bn5(self.conv5(out5)), inplace=True)

        out1b = F.relu(self.bn1b(self.conv1b(out1)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out2)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out3)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out4)), inplace=True)
        out5b = F.relu(self.bn5b(self.conv5b(out5)), inplace=True)

        out1d = F.relu(self.bn1d(self.conv1d(out1)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out2)), inplace=True)
        out3d = F.relu(self.bn3d(self.conv3d(out3)), inplace=True)
        out4d = F.relu(self.bn4d(self.conv4d(out4)), inplace=True)
        out5d = F.relu(self.bn5d(self.conv5d(out5)), inplace=True)

        return (out5b, out4b, out3b, out2b, out1b), (out5d, out4d, out3d, out2d, out1d)

    def initialize(self):
        weight_init(self)



# ==============================================================================================

class ME_Module(nn.Module):
    def __init__(self):
        super(ME_Module, self).__init__()
        self.edecoderg = Enhance_Decoder()
        self.edecodera = Enhance_Decoder()

    def forward(self, g1, a1, g2=[0, 0, 0, 0, 0],a2=[0, 0, 0, 0, 0],g3=[0, 0, 0, 0, 0],a3=[0, 0, 0, 0, 0]):
        outg = self.edecoderg(g1, g2, g3)
        outa = self.edecodera(a1, a2, a3)
        out = torch.cat([outg, outa], dim=1)

        return outg, outa, out

    def initialize(self):
        weight_init(self)

def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


class ME_Attention(nn.Module):
    def __init__(self, channels=64, r=4):
        super(ME_Attention, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei

    def initialize(self):
        # weight_init(self)
        pass



class Backbone_Encoder(nn.Module):
    def __init__(self):
        super(Backbone_Encoder, self).__init__()
        self.conv5g = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4g = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3g = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2g = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1g = nn.Sequential(nn.Conv2d(  64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv5a = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4a = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3a = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2a = nn.Sequential(nn.Conv2d( 256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1a = nn.Sequential(nn.Conv2d(  64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, out1, out2, out3, out4, out5):
        out1g, out2g, out3g, out4g, out5g = self.conv1g(out1), self.conv2g(out2), self.conv3g(out3), self.conv4g(out4), self.conv5g(out5)
        out1a, out2a, out3a, out4a, out5a = self.conv1a(out1), self.conv2a(out2), self.conv3a(out3), self.conv4a(out4), self.conv5a(out5)

        return (out5g, out4g, out3g, out2g, out1g), (out5a, out4a, out3a, out2a, out1a)

    def initialize(self):
        weight_init(self)


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        self.encoder = encoder

        self.bkbone_encoder = Backbone_Encoder()

        self.me_module = ME_Module()

        self.eencoder1 = Enhance_Encoder()
        self.eencoder2 = Enhance_Encoder()
        self.eencoder3 = Enhance_Encoder()


        self.linearg = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineara = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))


    def forward(self, x, phase='test'):
        out1, out2, out3, out4, out5 = self.encoder(x)

        # Backbone Encoder
        g1, a1 = self.bkbone_encoder(out1, out2, out3, out4, out5)

        outg1, outa1, out1 = self.me_module(list(reversed(g1)), list(reversed(a1)))
        out1 = F.interpolate(out1, size=g1[4].size()[2:], mode='bilinear')

        g2, a2 = self.eencoder1(out1)
        outg2, outa2, out2 = self.me_module(g1, a1, g2, a2)

        g3, a3 = self.eencoder2(out2)
        outg3, outa3, out3 = self.me_module(list(reversed(g1)), list(reversed(a1)),
                                            list(reversed(g3)), list(reversed(a3)))
        out3 = F.interpolate(out3, size=g1[4].size()[2:], mode='bilinear')

        g4, a4 = self.eencoder3(out3)
        outg4, outa4, out4 = self.me_module(g1, a1, g4, a4, g3, a3)

        #if shape is None:
        shape = x.size()[2:]
        out1 = F.interpolate(self.linear(out1), size=shape, mode='bilinear')
        outg1 = F.interpolate(self.linearg(outg1), size=shape, mode='bilinear')

        out2 = F.interpolate(self.linear(out2), size=shape, mode='bilinear')
        outg2 = F.interpolate(self.linearg(outg2), size=shape, mode='bilinear')

        out3 = F.interpolate(self.linear(out3), size=shape, mode='bilinear')
        outg3 = F.interpolate(self.linearg(outg3), size=shape, mode='bilinear')

        out4 = F.interpolate(self.linear(out4), size=shape, mode='bilinear')
        outg4 = F.interpolate(self.linearg(outg4), size=shape, mode='bilinear')

        
        out_dict = {}
        out_dict['sal'] = [out1, out2, out3, out4]
        out_dict['edge'] = [outg1, outg2, outg3, outg4]
        out_dict['final'] = out4

        return out_dict
