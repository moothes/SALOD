import torch
import torch.nn as nn
import torch.nn.functional as F

custom_config = {'base'      : {'strategy': 'adam_base',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }
         

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()

        self.config = config
        self.encoder = encoder
        if config['backbone'] == 'vgg':
            pass
        else:
            self.post1 = nn.Conv2d(feat[0],64,3,padding=1) # 16
            self.post2 = nn.Conv2d(feat[1],128,3,padding=1) # 16
            self.post3 = nn.Conv2d(feat[2],256,3,padding=1) # 16
            self.post4 = nn.Conv2d(feat[3],512,3,padding=1) # 16

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 5
        self.resb5_1 = BasicBlock(512,512)
        self.resb5_2 = BasicBlock(512,512)
        self.resb5_3 = BasicBlock(512,512) #14

        self.pool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        #stage 6
        self.resb6_1 = BasicBlock(512,512)
        self.resb6_2 = BasicBlock(512,512)
        self.resb6_3 = BasicBlock(512,512) #7

        ## -------------Bridge--------------

        #stage Bridge
        self.convbg_1 = nn.Conv2d(512,512,3,dilation=2, padding=2) # 7
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)
        self.convbg_m = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)
        self.convbg_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        ## -------------Decoder--------------

        #stage 6d
        self.conv6d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn6d_1 = nn.BatchNorm2d(512)
        self.relu6d_1 = nn.ReLU(inplace=True)

        self.conv6d_m = nn.Conv2d(512,512,3,dilation=2, padding=2)###
        self.bn6d_m = nn.BatchNorm2d(512)
        self.relu6d_m = nn.ReLU(inplace=True)

        self.conv6d_2 = nn.Conv2d(512,512,3,dilation=2, padding=2)
        self.bn6d_2 = nn.BatchNorm2d(512)
        self.relu6d_2 = nn.ReLU(inplace=True)

        #stage 5d
        self.conv5d_1 = nn.Conv2d(1024,512,3,padding=1) # 16
        self.bn5d_1 = nn.BatchNorm2d(512)
        self.relu5d_1 = nn.ReLU(inplace=True)

        self.conv5d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn5d_m = nn.BatchNorm2d(512)
        self.relu5d_m = nn.ReLU(inplace=True)

        self.conv5d_2 = nn.Conv2d(512,512,3,padding=1)
        self.bn5d_2 = nn.BatchNorm2d(512)
        self.relu5d_2 = nn.ReLU(inplace=True)

        #stage 4d
        self.conv4d_1 = nn.Conv2d(1024,512,3,padding=1) # 32
        self.bn4d_1 = nn.BatchNorm2d(512)
        self.relu4d_1 = nn.ReLU(inplace=True)

        self.conv4d_m = nn.Conv2d(512,512,3,padding=1)###
        self.bn4d_m = nn.BatchNorm2d(512)
        self.relu4d_m = nn.ReLU(inplace=True)

        self.conv4d_2 = nn.Conv2d(512,256,3,padding=1)
        self.bn4d_2 = nn.BatchNorm2d(256)
        self.relu4d_2 = nn.ReLU(inplace=True)

        #stage 3d
        self.conv3d_1 = nn.Conv2d(512,256,3,padding=1) # 64
        self.bn3d_1 = nn.BatchNorm2d(256)
        self.relu3d_1 = nn.ReLU(inplace=True)

        self.conv3d_m = nn.Conv2d(256,256,3,padding=1)###
        self.bn3d_m = nn.BatchNorm2d(256)
        self.relu3d_m = nn.ReLU(inplace=True)

        self.conv3d_2 = nn.Conv2d(256,128,3,padding=1)
        self.bn3d_2 = nn.BatchNorm2d(128)
        self.relu3d_2 = nn.ReLU(inplace=True)

        #stage 2d

        self.conv2d_1 = nn.Conv2d(256,128,3,padding=1) # 128
        self.bn2d_1 = nn.BatchNorm2d(128)
        self.relu2d_1 = nn.ReLU(inplace=True)

        self.conv2d_m = nn.Conv2d(128,128,3,padding=1)###
        self.bn2d_m = nn.BatchNorm2d(128)
        self.relu2d_m = nn.ReLU(inplace=True)

        self.conv2d_2 = nn.Conv2d(128,64,3,padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.relu2d_2 = nn.ReLU(inplace=True)

        #stage 1d
        self.conv1d_1 = nn.Conv2d(128,64,3,padding=1) # 256
        self.bn1d_1 = nn.BatchNorm2d(64)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.conv1d_m = nn.Conv2d(64,64,3,padding=1)###
        self.bn1d_m = nn.BatchNorm2d(64)
        self.relu1d_m = nn.ReLU(inplace=True)

        self.conv1d_2 = nn.Conv2d(64,64,3,padding=1)
        self.bn1d_2 = nn.BatchNorm2d(64)
        self.relu1d_2 = nn.ReLU(inplace=True)

        ## -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')###
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        ## -------------Side Output--------------
        self.outconvb = nn.Conv2d(512,1,3,padding=1)
        self.outconv6 = nn.Conv2d(512,1,3,padding=1)
        self.outconv5 = nn.Conv2d(512,1,3,padding=1)
        self.outconv4 = nn.Conv2d(256,1,3,padding=1)
        self.outconv3 = nn.Conv2d(128,1,3,padding=1)
        self.outconv2 = nn.Conv2d(64,1,3,padding=1)
        self.outconv1 = nn.Conv2d(64,1,3,padding=1)

        ## -------------Refine Module-------------
        self.refunet = RefUnet(1,64)


    def forward(self, x, phase='test'):
        x_size = x.size()[-2:]
        hx = x

        ## -------------Encoder-------------
        h1, h2, h3, h4, _ = self.encoder(hx)
        if self.config['backbone'] != 'vgg':
            
            h1 = self.post1(h1)
            h2 = self.post2(h2)
            h3 = self.post3(h3)
            h4 = self.post4(h4)
            
        hx = self.pool4(h4) # 16

        hx = self.resb5_1(hx)
        hx = self.resb5_2(hx)
        h5 = self.resb5_3(hx)

        hx = self.pool5(h5) # 8

        hx = self.resb6_1(hx)
        hx = self.resb6_2(hx)
        h6 = self.resb6_3(hx)

        ## -------------Bridge-------------
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(h6))) # 8
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))

        ## -------------Decoder-------------

        hx = self.relu6d_1(self.bn6d_1(self.conv6d_1(torch.cat((hbg,h6),1))))
        hx = self.relu6d_m(self.bn6d_m(self.conv6d_m(hx)))
        hd6 = self.relu6d_2(self.bn6d_2(self.conv6d_2(hx)))

        hx = self.upscore2(hd6) # 8 -> 16

        hx = self.relu5d_1(self.bn5d_1(self.conv5d_1(torch.cat((hx,h5),1))))
        hx = self.relu5d_m(self.bn5d_m(self.conv5d_m(hx)))
        hd5 = self.relu5d_2(self.bn5d_2(self.conv5d_2(hx)))

        hx = self.upscore2(hd5) # 16 -> 32

        hx = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((hx,h4),1))))
        hx = self.relu4d_m(self.bn4d_m(self.conv4d_m(hx)))
        hd4 = self.relu4d_2(self.bn4d_2(self.conv4d_2(hx)))

        hx = self.upscore2(hd4) # 32 -> 64

        hx = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((hx,h3),1))))
        hx = self.relu3d_m(self.bn3d_m(self.conv3d_m(hx)))
        hd3 = self.relu3d_2(self.bn3d_2(self.conv3d_2(hx)))

        hx = self.upscore2(hd3) # 64 -> 128

        hx = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hx,h2),1))))
        hx = self.relu2d_m(self.bn2d_m(self.conv2d_m(hx)))
        hd2 = self.relu2d_2(self.bn2d_2(self.conv2d_2(hx)))

        hx = self.upscore2(hd2) # 128 -> 256

        hx = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hx,h1),1))))
        hx = self.relu1d_m(self.bn1d_m(self.conv1d_m(hx)))
        hd1 = self.relu1d_2(self.bn1d_2(self.conv1d_2(hx)))

        ## -------------Side Output-------------
        db = self.outconvb(hbg)
        db = nn.functional.interpolate(db, size=x_size, mode='bilinear')
        #db = self.upscore6(db) # 8->256

        d6 = self.outconv6(hd6)
        d6 = nn.functional.interpolate(d6, size=x_size, mode='bilinear')
        #d6 = self.upscore6(d6) # 8->256

        d5 = self.outconv5(hd5)
        d5 = nn.functional.interpolate(d5, size=x_size, mode='bilinear')
        #d5 = self.upscore5(d5) # 16->256

        d4 = self.outconv4(hd4)
        d4 = nn.functional.interpolate(d4, size=x_size, mode='bilinear')
        #d4 = self.upscore4(d4) # 32->256

        d3 = self.outconv3(hd3)
        d3 = nn.functional.interpolate(d3, size=x_size, mode='bilinear')
        #d3 = self.upscore3(d3) # 64->256

        d2 = self.outconv2(hd2)
        d2 = nn.functional.interpolate(d2, size=x_size, mode='bilinear')
        #d2 = self.upscore2(d2) # 128->256

        d1 = self.outconv1(hd1) # 256
        d1 = nn.functional.interpolate(d1, size=x_size, mode='bilinear')

        ## -------------Refine Module-------------
        dout = self.refunet(d1) # 256
        #dout = nn.functional.interpolate(dout, size=x_size, mode='bilinear')

        out_dict = {}
        out_dict['final'] = dout
        out_dict['sal'] = [dout, d1, d2, d3, d4, d5, d6, db]
        return out_dict
