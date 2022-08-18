import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0,0,0,0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out1 = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out2 = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out3
    
    def initialize(self):
        weight_init(self)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b   = nn.BatchNorm2d(64)

        self.conv1d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d   = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d   = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d   = nn.BatchNorm2d(64)
        self.conv4d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d   = nn.BatchNorm2d(64)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)

        out1b = F.relu(self.bn1b(self.conv1b(out1)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out2)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out3)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out4)), inplace=True)

        out1d = F.relu(self.bn1d(self.conv1d(out1)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out2)), inplace=True)
        out3d = F.relu(self.bn3d(self.conv3d(out3)), inplace=True)
        out4d = F.relu(self.bn4d(self.conv4d(out4)), inplace=True)
        return (out4b, out3b, out2b, out1b), (out4d, out3d, out2d, out1d)

    def initialize(self):
        weight_init(self)


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        self.encoder  = encoder
        self.conv5b   = nn.Sequential(nn.Conv2d(feat[4], 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b   = nn.Sequential(nn.Conv2d(feat[3], 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b   = nn.Sequential(nn.Conv2d(feat[2], 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b   = nn.Sequential(nn.Conv2d(feat[1], 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.conv5d   = nn.Sequential(nn.Conv2d(feat[4], 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d   = nn.Sequential(nn.Conv2d(feat[3], 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d   = nn.Sequential(nn.Conv2d(feat[2], 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d   = nn.Sequential(nn.Conv2d(feat[1], 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.encoder1  = Encoder()
        self.decoderb = Decoder()
        self.decoderd = Decoder()
        self.linearb  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.lineard  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x, phase='test'):
        shape = x.size()[2:]
        out1, out2, out3, out4, out5 = self.encoder(x)
        out2b, out3b, out4b, out5b   = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d   = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        outb1 = self.decoderb([out5b, out4b, out3b, out2b])
        outd1 = self.decoderd([out5d, out4d, out3d, out2d])
        out1  = torch.cat([outb1, outd1], dim=1)
        outb2, outd2 = self.encoder1(out1)
        outb2 = self.decoderb([out5b, out4b, out3b, out2b], outb2)
        outd2 = self.decoderd([out5d, out4d, out3d, out2d], outd2)
        out2  = torch.cat([outb2, outd2], dim=1)

        out1  = F.interpolate(self.linear(out1),   size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb(outb1), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard(outd1), size=shape, mode='bilinear')

        out2  = F.interpolate(self.linear(out2),   size=shape, mode='bilinear')
        outb2 = F.interpolate(self.linearb(outb2), size=shape, mode='bilinear')
        outd2 = F.interpolate(self.lineard(outd2), size=shape, mode='bilinear')
        #return outb1, outd1, out1, outb2, outd2, out2
        
        OutDict = {}
        OutDict['final'] = out2
        OutDict['body'] = [outb1, outb2]
        OutDict['detail'] = [outd1, outd2]
        OutDict['sal'] = [out1, out2]
        return OutDict



