import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet50


###ECCV2020 Suppress and Balance: A Simple Gated Network for Salient Object Detection
class FoldConv_aspp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 win_size=3, win_dilation=1, win_padding=0):
        super(FoldConv_aspp, self).__init__()
        #down_C = in_channel // 8
        self.down_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3,padding=1),nn.BatchNorm2d(out_channel),
             nn.PReLU())
        self.win_size = win_size
        self.unfold = nn.Unfold(win_size, win_dilation, win_padding, win_size)
        fold_C = out_channel * win_size * win_size
        down_dim = fold_C // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim,kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size, stride, padding, dilation, groups),
            nn.BatchNorm2d(down_dim),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d( down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(fold_C, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU()  #如果batch=1 ，进行batchnorm会有问题
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, fold_C, kernel_size=1), nn.BatchNorm2d(fold_C), nn.PReLU()
        )

        # self.fold = nn.Fold(out_size, win_size, win_dilation, win_padding, win_size)

        self.up_conv = nn.Conv2d(out_channel, out_channel, 1)

    def forward(self, in_feature):
        N, C, H, W = in_feature.size()
        in_feature = self.down_conv(in_feature)
        in_feature = self.unfold(in_feature)
        in_feature = in_feature.view(in_feature.size(0), in_feature.size(1),
                                     H // self.win_size, W // self.win_size)
        in_feature1 = self.conv1(in_feature)
        in_feature2 = self.conv2(in_feature)
        in_feature3 = self.conv3(in_feature)
        in_feature4 = self.conv4(in_feature)
        in_feature5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(in_feature, 1)), size=in_feature.size()[2:], mode='bilinear')
        in_feature = self.fuse(torch.cat((in_feature1, in_feature2, in_feature3,in_feature4,in_feature5), 1))
        in_feature = in_feature.reshape(in_feature.size(0), in_feature.size(1), -1)


        in_feature = F.fold(input=in_feature, output_size=H, kernel_size=2, dilation=1, padding=0, stride=2)
        in_feature = self.up_conv(in_feature)
        return in_feature

        
        
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

class decoder(nn.Module):
    def __init__(self, config, feat):
        super(Network, self).__init__()
        
        #print(feat)
        #resnet [64, 128, 256, 512, 512]
        #vgg [64, 256, 512, 1024, 2048]

        ################################Gate#######################################
        self.attention_feature5 = nn.Sequential(nn.Conv2d(feat[0]+64, 2, kernel_size=3, padding=1))
        self.attention_feature4 = nn.Sequential(nn.Conv2d(feat[1]+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature3 = nn.Sequential(nn.Conv2d(feat[2]+128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                nn.Conv2d(128, 2, kernel_size=3, padding=1))
        self.attention_feature2 = nn.Sequential(nn.Conv2d(feat[3]+256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                                nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature1 = nn.Sequential(nn.Conv2d(feat[4]+512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU(),
                                                 nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                 nn.Conv2d(128, 2, kernel_size=3, padding=1))
        ###############################Transition Layer########################################
        self.dem1 = nn.Sequential(FoldConv_aspp(in_channel=feat[4],
                      out_channel=512,
                      out_size=384 // 16,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2,
                      win_size=2,
                      win_padding=0,
        ), nn.BatchNorm2d(512), nn.PReLU())
        self.dem2 = nn.Sequential(nn.Conv2d(feat[3], 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.dem3 = nn.Sequential(nn.Conv2d(feat[2], 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.dem4 = nn.Sequential(nn.Conv2d(feat[1], 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.dem5 = nn.Sequential(nn.Conv2d(feat[0], 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        ################################FPN branch#######################################
        self.output1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        ################################Parallel branch#######################################
        self.out_res = nn.Sequential(nn.Conv2d(512+256+128+64+64+1, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                     nn.Conv2d(256, 1, kernel_size=3, padding=1))
        #######################################################################
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True


    def forward(self, xs, phase='test'):
        #input = x
        #B,_,_,_ = input.size()
        E1,E2,E3,E4,E5 = xs
        
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Gated FPN#######################################
        G5 = self.attention_feature1(torch.cat((E5, T5), 1))
        G5 = F.adaptive_avg_pool2d(F.sigmoid(G5), 1)
        D5 = self.output1(G5[:, 0, :, :].unsqueeze(1).repeat(1, 512, 1, 1) * T5)

        G4 = self.attention_feature2(torch.cat((E4,F.upsample(D5, size=E4.size()[2:], mode='bilinear')),1))
        G4 = F.adaptive_avg_pool2d(F.sigmoid(G4),1)
        D4 = self.output2(F.upsample(D5, size=E4.size()[2:], mode='bilinear')+G4[:, 0,:,:].unsqueeze(1).repeat(1,256,1,1)*T4)

        G3 = self.attention_feature3(torch.cat((E3,F.upsample(D4, size=E3.size()[2:], mode='bilinear')),1))
        G3 = F.adaptive_avg_pool2d(F.sigmoid(G3),1)
        D3 = self.output3(F.upsample(D4, size=E3.size()[2:], mode='bilinear')+G3[:, 0,:,:].unsqueeze(1).repeat(1,128,1,1)*T3)

        G2 = self.attention_feature4(torch.cat((E2,F.upsample(D3, size=E2.size()[2:], mode='bilinear')),1))
        G2 = F.adaptive_avg_pool2d(F.sigmoid(G2),1)
        D2 = self.output4(F.upsample(D3, size=E2.size()[2:], mode='bilinear')+G2[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T2)

        G1 = self.attention_feature5(torch.cat((E1,F.upsample(D2, size=E1.size()[2:], mode='bilinear')),1))
        G1 = F.adaptive_avg_pool2d(F.sigmoid(G1),1)
        D1 = self.output5(F.upsample(D2, size=E1.size()[2:], mode='bilinear')+G1[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T1)
        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_res = self.out_res(torch.cat((D1,F.upsample(G5[:, 1,:,:].unsqueeze(1).repeat(1,512,1,1)*T5,size=E1.size()[2:], mode='bilinear'),F.upsample(G4[:, 1,:,:].unsqueeze(1).repeat(1,256,1,1)*T4,size=E1.size()[2:], mode='bilinear'),F.upsample(G3[:, 1,:,:].unsqueeze(1).repeat(1,128,1,1)*T3,size=E1.size()[2:], mode='bilinear'),F.upsample(G2[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T2,size=E1.size()[2:], mode='bilinear'),F.upsample(G1[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T1,size=E1.size()[2:], mode='bilinear')),1))
        output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        output_fpn = F.upsample(D1, size=input.size()[2:], mode='bilinear')
        pre_sal = output_fpn + output_res
        #######################################################################
        #if self.training:
        #    return output_fpn, pre_sal
        # return F.sigmoid(pre_sal)
        #return pre_sal
        OutDict = {}
        OutDict['final'] = pre_sal
        OutDict['sal'] = [pre_sal, output_fpn]
        
        return OutDict


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        
        self.encoder = encoder

        #print(feat)
        #resnet [64, 128, 256, 512, 512]
        #vgg [64, 256, 512, 1024, 2048]

        ################################Gate#######################################
        self.attention_feature5 = nn.Sequential(nn.Conv2d(feat[0]+64, 2, kernel_size=3, padding=1))
        self.attention_feature4 = nn.Sequential(nn.Conv2d(feat[1]+64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature3 = nn.Sequential(nn.Conv2d(feat[2]+128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                nn.Conv2d(128, 2, kernel_size=3, padding=1))
        self.attention_feature2 = nn.Sequential(nn.Conv2d(feat[3]+256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                                nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
                                                nn.Conv2d(64, 2, kernel_size=3, padding=1))
        self.attention_feature1 = nn.Sequential(nn.Conv2d(feat[4]+512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU(),
                                                 nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
                                                 nn.Conv2d(128, 2, kernel_size=3, padding=1))
        ###############################Transition Layer########################################
        self.dem1 = nn.Sequential(FoldConv_aspp(in_channel=feat[4],
                      out_channel=512,
                      out_size=384 // 16,
                      kernel_size=3,
                      stride=1,
                      padding=2,
                      dilation=2,
                      win_size=2,
                      win_padding=0,
        ), nn.BatchNorm2d(512), nn.PReLU())
        self.dem2 = nn.Sequential(nn.Conv2d(feat[3], 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.dem3 = nn.Sequential(nn.Conv2d(feat[2], 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.dem4 = nn.Sequential(nn.Conv2d(feat[1], 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.dem5 = nn.Sequential(nn.Conv2d(feat[0], 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        ################################FPN branch#######################################
        self.output1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        ################################Parallel branch#######################################
        self.out_res = nn.Sequential(nn.Conv2d(512+256+128+64+64+1, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
                                     nn.Conv2d(256, 1, kernel_size=3, padding=1))
        #######################################################################
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True


    def forward(self, x, phase='test'):
        input = x
        B,_,_,_ = input.size()
        E1,E2,E3,E4,E5 = self.encoder(input)
        
        ################################Transition Layer#######################################
        T5 = self.dem1(E5)
        T4 = self.dem2(E4)
        T3 = self.dem3(E3)
        T2 = self.dem4(E2)
        T1 = self.dem5(E1)
        ################################Gated FPN#######################################
        G5 = self.attention_feature1(torch.cat((E5, T5), 1))
        G5 = F.adaptive_avg_pool2d(F.sigmoid(G5), 1)
        D5 = self.output1(G5[:, 0, :, :].unsqueeze(1).repeat(1, 512, 1, 1) * T5)

        G4 = self.attention_feature2(torch.cat((E4,F.upsample(D5, size=E4.size()[2:], mode='bilinear')),1))
        G4 = F.adaptive_avg_pool2d(F.sigmoid(G4),1)
        D4 = self.output2(F.upsample(D5, size=E4.size()[2:], mode='bilinear')+G4[:, 0,:,:].unsqueeze(1).repeat(1,256,1,1)*T4)

        G3 = self.attention_feature3(torch.cat((E3,F.upsample(D4, size=E3.size()[2:], mode='bilinear')),1))
        G3 = F.adaptive_avg_pool2d(F.sigmoid(G3),1)
        D3 = self.output3(F.upsample(D4, size=E3.size()[2:], mode='bilinear')+G3[:, 0,:,:].unsqueeze(1).repeat(1,128,1,1)*T3)

        G2 = self.attention_feature4(torch.cat((E2,F.upsample(D3, size=E2.size()[2:], mode='bilinear')),1))
        G2 = F.adaptive_avg_pool2d(F.sigmoid(G2),1)
        D2 = self.output4(F.upsample(D3, size=E2.size()[2:], mode='bilinear')+G2[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T2)

        G1 = self.attention_feature5(torch.cat((E1,F.upsample(D2, size=E1.size()[2:], mode='bilinear')),1))
        G1 = F.adaptive_avg_pool2d(F.sigmoid(G1),1)
        D1 = self.output5(F.upsample(D2, size=E1.size()[2:], mode='bilinear')+G1[:, 0,:,:].unsqueeze(1).repeat(1,64,1,1)*T1)
        ################################Gated Parallel&Dual branch residual fuse#######################################
        output_fpn = F.upsample(D1, size=input.size()[2:], mode='bilinear')
        output_res = self.out_res(torch.cat((D1,F.upsample(G5[:, 1,:,:].unsqueeze(1).repeat(1,512,1,1)*T5,size=E1.size()[2:], mode='bilinear'),F.upsample(G4[:, 1,:,:].unsqueeze(1).repeat(1,256,1,1)*T4,size=E1.size()[2:], mode='bilinear'),F.upsample(G3[:, 1,:,:].unsqueeze(1).repeat(1,128,1,1)*T3,size=E1.size()[2:], mode='bilinear'),F.upsample(G2[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T2,size=E1.size()[2:], mode='bilinear'),F.upsample(G1[:, 1,:,:].unsqueeze(1).repeat(1,64,1,1)*T1,size=E1.size()[2:], mode='bilinear')),1))
        output_res =  F.upsample(output_res,size=input.size()[2:], mode='bilinear')
        pre_sal = output_fpn + output_res
        #######################################################################
        #if self.training:
        #    return output_fpn, pre_sal
        # return F.sigmoid(pre_sal)
        #return pre_sal
        OutDict = {}
        OutDict['final'] = pre_sal
        OutDict['sal'] = [pre_sal, output_fpn]
        
        return OutDict
