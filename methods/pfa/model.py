#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet

def concatenate(inputs,axis):
    h, w = 0, 0
    for i in inputs:
        if i.shape[2] > h:
            h = i.shape[2]
        if i.shape[3] > w:
            w = i.shape[3]
    upsample = []
    for i in inputs:
        upsample.append(nn.UpsamplingBilinear2d(size=(h, w))(i))
    return torch.cat(upsample,axis)

class decoder(nn.Module):
    def __init__(self, feat, config):
        super(decoder, self).__init__()
        #params
        self.dropout = config['dropout']
        self.with_CA = config['with_CA']
        self.with_SA = config['with_SA']
        
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        #c1,c2
        self.conv14 = nn.Conv2d(feat[0],64,(3,3),padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64,affine=False)
        self.conv15 = nn.Conv2d(feat[1],64,(3,3),padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64,affine=False)
        #c12
        self.conv16 = nn.Conv2d(128,64,(3,3),padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64,affine=False)
        #cfe3
        self.conv17 = nn.Conv2d(feat[2],32,(1,1),padding=0)
        self.conv18 = nn.Conv2d(feat[2],32,(3,3),dilation=3,padding=3)
        self.conv19 = nn.Conv2d(feat[2],32,(3,3),dilation=5,padding=5)
        self.conv20 = nn.Conv2d(feat[2],32,(3,3),dilation=7,padding=7)
        self.bn4 = nn.BatchNorm2d(num_features=128,affine=False)
        #cfe4
        self.conv21 = nn.Conv2d(feat[3],32,(1,1),padding=0)
        self.conv22 = nn.Conv2d(feat[3],32,(3,3),dilation=3,padding=3)
        self.conv23 = nn.Conv2d(feat[3],32,(3,3),dilation=5,padding=5)
        self.conv24 = nn.Conv2d(feat[3],32,(3,3),dilation=7,padding=7)
        self.bn5 = nn.BatchNorm2d(num_features=128,affine=False)
        #cfe5
        self.conv25 = nn.Conv2d(feat[4],32,(1,1),padding=0)
        self.conv26 = nn.Conv2d(feat[4],32,(3,3),dilation=3,padding=3)
        self.conv27 = nn.Conv2d(feat[4],32,(3,3),dilation=5,padding=5)
        self.conv28 = nn.Conv2d(feat[4],32,(3,3),dilation=7,padding=7)
        self.bn6 = nn.BatchNorm2d(num_features=128,affine=False)
        #channel wise attention
        self.linear1 = nn.Linear(384,96)
        self.linear2 = nn.Linear(96,384)
        self.conv29 = nn.Conv2d(384,64,(1,1),padding=0)
        self.bn7 = nn.BatchNorm2d(num_features=64,affine=False)
        #SpatialAttention
        self.conv30 = nn.Conv2d(64,32,(1,9),padding=(0,4))
        self.bn8 = nn.BatchNorm2d(num_features=32,affine=False)
        self.conv31 = nn.Conv2d(32,1,(9,1),padding=(4,0))
        self.bn9 = nn.BatchNorm2d(num_features=1,affine=False)
        self.conv32 = nn.Conv2d(64,32,(9,1),padding=(4,0))
        self.bn10 = nn.BatchNorm2d(num_features=32,affine=False)
        self.conv33 = nn.Conv2d(32,1,(1,9),padding=(0,4))
        self.bn11 = nn.BatchNorm2d(num_features=1,affine=False)
        #final conv
        self.conv34 = nn.Conv2d(128,1,(3,3),padding=1)

            
    def forward(self, xs, w, h):
        C1, C2, C3, C4, C5 = xs
        C1 = self.conv14(C1)
        C1 = self.relu(self.bn1(C1))
        C2 = self.conv15(C2)
        C2 = self.relu(self.bn2(C2))
        C12 = concatenate([C1,C2],1)	#C12: [-1, 64+128, h, w]
        C12 = self.conv16(C12)
        C12 = self.relu(self.bn3(C12))	#C12: [-1, 64, h, w]
        C3_cfe = self.relu(self.bn4(concatenate([self.conv17(C3),self.conv18(C3),self.conv19(C3),self.conv20(C3)],1)))
        C4_cfe = self.relu(self.bn5(concatenate([self.conv21(C4),self.conv22(C4),self.conv23(C4),self.conv24(C4)],1)))
        C5_cfe = self.relu(self.bn6(concatenate([self.conv25(C5),self.conv26(C5),self.conv27(C5),self.conv28(C5)],1)))
        C345 = concatenate([C3_cfe,C4_cfe,C5_cfe],1)	#C345: [-1, 32*4*3, h/4, w/4]
        if self.with_CA:
            _h, _w = C345.shape[2:]
            CA = nn.AvgPool2d(_h*_w)(C345).view(-1,384)
            CA = self.linear1(CA)
            CA = self.linear2(CA).view((-1,384,1,1)).repeat([1,1,_h,_w])
            C345 = CA * C345
        C345 = self.conv29(C345)
        C345 = self.relu(self.bn7(C345))	#C345: [-1, 64, h/4, w/4]
        C345 = nn.UpsamplingBilinear2d(size=(h, w))(C345)	#C345: [-1, 64, h, w]
        if self.with_SA:
            attention1 = self.relu(self.bn8(self.conv30(C345)))	#[-1, 32, h, w]
            attention1 = self.relu(self.bn9(self.conv31(attention1)))	#[-1, 1, h, w]
            attention2 = self.relu(self.bn10(self.conv32(C345)))#[-1, 32, h, w]
            attention2 = self.relu(self.bn11(self.conv33(attention2)))	#[-1, 1, h, w]
            SA = attention1 + attention2
            SA = self.sigmoid(SA)	#[-1, 1, h, w]
            SA = SA.repeat([1,64,1,1])
            C12 = nn.UpsamplingBilinear2d(size=(h, w))(C12)	#C345: [-1, 64, h, w]
            C12 = SA * C12	#[-1, 64, h, w]
        fea = torch.cat([C12,C345],1)	#[-1, 128, h, w]
        x = self.conv34(fea)
        
        out_dict = {}
        out_dict['final'] = x
        return out_dict

class Network(nn.Module):
    #def __init__(self, dropout=True, with_CA=True, with_SA=True, drop_rate=0.3):
    def __init__(self, config, encoder, feat):
        super(Network,self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder(feat, config)

    def forward(self, x, phase='test'):
        #x: [batch_size, channel=3, h, w]
        h, w = x.shape[2:]
        xs = self.encoder(x)
        res = self.decoder(xs, w, h)
        return res
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.randn([1,3,256,256])
    print('original shape:',t.shape)
    model = Model().to(device)
    t = Variable(t.to(device))
    output = model(t)
    print('model output shape:',output.shape)
