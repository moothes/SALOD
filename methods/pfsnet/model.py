import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet50

'''
class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        self.encoder = encoder
        self.decoder = nn.Conv2d(feat[-1], 1, 1)
    
    def forward(self, x, phase='test'):
        # phase: enable different operations in training/testing phases. Useful in several networks.
        x_size = x.size()[2:]
        enc_feats = self.encoder(x)
        x = self.decoder(enc_feats[-1])
        pred = nn.functional.interpolate(x, size=x_size, mode='bilinear')
        
        # All output are save in a Dict, while the key of final prediction is 'final'
        OutDict = {}
        OutDict['final'] = pred
        return OutDict
'''


class Rblock(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(Rblock,self).__init__()
        self.squeeze1 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,stride=1,dilation=2,padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.convg = nn.Conv2d(128,128,1)
        self.sftmax = nn.Softmax(dim=1)    
        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB   = nn.BatchNorm2d(64)

    def forward(self, x,z):
        z = self.squeeze1(x) if z is None else self.squeeze1(x+z)
        x = self.squeeze2(x)
        x = torch.cat((x,z),1)
        y = self.convg(self.gap(x))       
        x = torch.mul(self.sftmax(y)*y.shape[1],x)     
        x = F.relu(self.bnAB(self.convAB(x)),inplace=True)        
        return x,z
    def initialize(self):
        weight_init(self)
        print("Rblock module init")

class Yblock(nn.Module):
    def __init__(self):
        super(Yblock,self).__init__()

        self.convA1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnA1   = nn.BatchNorm2d(64)
        
        self.convB1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bnB1   = nn.BatchNorm2d(64)
        
        self.convAB = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bnAB   = nn.BatchNorm2d(64)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.convg = nn.Conv2d(128,128,1)
        self.sftmax = nn.Softmax(dim=1)
      #  self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x,y):
        if x.size()[2:] != y.size()[2:]:
            y = F.interpolate(y, size=x.size()[2:], mode='bilinear',align_corners=True)
        fuze = torch.mul(x,y)
        y = F.relu(self.bnB1(self.convB1(fuze+y)),inplace=True)
        x = F.relu(self.bnA1(self.convA1(fuze+x)),inplace=True)
        x = torch.cat((x,y),1)
        y = self.convg(self.gap(x))
        x = torch.mul(self.sftmax(y)*y.shape[1],x)     
     #   x = self.dropout(x)
        x = F.relu(self.bnAB(self.convAB(x)),inplace=True)
        return x

    def initialize(self):
        weight_init(self)
        print("Yblock module init")

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        self.encoder   = encoder
        self.squeeze5 = Rblock(feat[4],64)
        self.squeeze4 = Rblock(feat[3],64)
        self.squeeze3 = Rblock(feat[2],64)
        self.squeeze2 = Rblock(feat[1],64)
        self.squeeze1 = Rblock(feat[0],64)

        self.conv1 = nn.Conv2d(64, feat[3], kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(feat[3])

        self.conv2 = nn.Conv2d(64, feat[2], kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(feat[2])

        self.conv3 = nn.Conv2d(64, feat[1], kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(feat[1])

        self.conv4 = nn.Conv2d(64, feat[0], kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(feat[0])

        self.Y11=Yblock()
        self.Y12=Yblock()
        self.Y13=Yblock()
        self.Y14=Yblock()

        self.Y21=Yblock()
        self.Y22=Yblock()
        self.Y23=Yblock()

        self.Y31=Yblock()
        self.Y32=Yblock()

        self.Y41=Yblock()

        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)


    def forward(self, x, phase='test'):
        shape = x.size()[2:]

        s1,s2,s3,s4,s5 = self.encoder(x)

        s5,z=self.squeeze5(s5,None)
        z= F.interpolate(z, size=s4.size()[2:],mode='bilinear',align_corners=True)
        z=F.relu(self.bn1(self.conv1(z)))
        s4,z=self.squeeze4(s4,z)
        z= F.interpolate(z, size=s3.size()[2:],mode='bilinear',align_corners=True)
        z=F.relu(self.bn2(self.conv2(z)))
        s3,z=self.squeeze3(s3,z)
        z= F.interpolate(z, size=s2.size()[2:],mode='bilinear',align_corners=True)
        z=F.relu(self.bn3(self.conv3(z)))
        s2,z=self.squeeze2(s2,z)
        z= F.interpolate(z, size=s1.size()[2:],mode='bilinear',align_corners=True)
        z=F.relu(self.bn4(self.conv4(z)))
        s1,z=self.squeeze1(s1,z)

        s5 = self.Y14(s4,s5)
        s4 = self.Y13(s3,s4)
        s3 = self.Y12(s2,s3)
        s2 = self.Y11(s1,s2)

        s2 = self.Y21(s2,s3)
        s3 = self.Y22(s3,s4)
        s4 = self.Y23(s4,s5)

        s4 = self.Y32(s3,s4)
        s3 = self.Y31(s2,s3)

        s3 = self.Y41(s3,s4)

        #shape = x.size()[2:] if shape is None else shape
        p1 = F.interpolate(self.linearp1(s3), size=shape, mode='bilinear',align_corners=True)
        del s1,s2,s3,s4,s5,z          
        torch.cuda.empty_cache()
        OutDict = {}
        OutDict['sal'] = [p1, ]
        OutDict['final'] = p1
        return OutDict

