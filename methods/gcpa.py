import torch
from torch import nn
from torch.nn import functional as F


custom_config = {'base'      : {'strategy': 'base_adam',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }
         

class CA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(CA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256
        down = down.mean(dim=(2,3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return left * down


""" Self Refinement Module """
class SRM(nn.Module):
    def __init__(self, in_channel):
        super(SRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)



""" Feature Interweaved Aggregation Module """
class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(FAM, self).__init__()
        #self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256*3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)


    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True) #256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True) #256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_1 * left, inplace=True)

        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')
        z3 = F.relu(down_2 * left, inplace=True)

        out = torch.cat((z1, z2, z3), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)



class SA(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(SA, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_down, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down_1 = self.conv2(down) #wb
        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')
        w,b = down_1[:,:256,:,:], down_1[:,256:,:,:]

        return F.relu(w*left+b, inplace=True)

class decoder(nn.Module):
    def __init__(self, config, feat, inc=256):
        super(decoder, self).__init__()
        
        self.ca45    = CA(feat[-1], feat[-1])
        self.ca35    = CA(feat[-1], feat[-1])
        self.ca25    = CA(feat[-1], feat[-1])
        self.ca55    = CA(inc, feat[-1])
        self.sa55   = SA(feat[-1], feat[-1])

        self.fam45   = FAM(feat[-2], inc, inc)
        self.fam34   = FAM(feat[-3], inc, inc)
        self.fam23   = FAM(feat[-4], inc, inc)

        self.srm5    = SRM(inc)
        self.srm4    = SRM(inc)
        self.srm3    = SRM(inc)
        self.srm2    = SRM(inc)

        self.linear5 = nn.Conv2d(inc, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(inc, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(inc, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(inc, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, xs, x_size):
        out1, out2, out3, out4, out5_ = xs
        # GCF
        out4_a = self.ca45(out5_, out5_)
        out3_a = self.ca35(out5_, out5_)
        out2_a = self.ca25(out5_, out5_)
        # HA
        out5_a = self.sa55(out5_, out5_)
        out5 = self.ca55(out5_a, out5_)
        # out
        out5 = self.srm5(out5)
        out4 = self.srm4(self.fam45(out4, out5, out4_a))
        out3 = self.srm3(self.fam34(out3, out4, out3_a))
        out2 = self.srm2(self.fam23(out2, out3, out2_a))
        # we use bilinear interpolation instead of transpose convolution
        out5  = F.interpolate(self.linear5(out5), size=x_size[2:], mode='bilinear')
        out4  = F.interpolate(self.linear4(out4), size=x_size[2:], mode='bilinear')
        out3  = F.interpolate(self.linear3(out3), size=x_size[2:], mode='bilinear')
        out2  = F.interpolate(self.linear2(out2), size=x_size[2:], mode='bilinear')
        
        out_dict = {}
        out_dict['final'] = out2
        out_dict['sal'] = [out2, out3, out4, out5]
        return out_dict

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        self.config   = config
        self.encoder  = encoder

        self.decoder = decoder(config, feat, 256)

    def forward(self, x, phase='test'):
        xs = self.encoder(x)
        out_dict = self.decoder(xs, x.size())
        return out_dict
