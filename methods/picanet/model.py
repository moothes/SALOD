import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable

from base.encoder.vgg import vgg
from base.encoder.resnet import resnet


def make_layers(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class DecoderCell(nn.Module):
    def __init__(self, size, in_channel, out_channel, mode):
        super(DecoderCell, self).__init__()
        self.bn_en = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'G':
            self.picanet = PicanetG(size, in_channel)
        elif mode == 'L':
            self.picanet = PicanetL(in_channel)
        elif mode == 'C':
            self.picanet = None
        else:
            assert 0
        if not mode == 'C':
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        else:
            self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0)

    def forward(self, *input):
        assert len(input) <= 2
        if input[1] is None:
            en = input[0]
            dec = input[0]
        else:
            en = input[0]
            dec = input[1]

        if dec.size()[2] * 2 == en.size()[2]:
            dec = F.interpolate(dec, scale_factor=2, mode='bilinear', align_corners=True)
        elif dec.size()[2] != en.size()[2]:
            assert 0
        en = self.bn_en(en)
        en = F.relu(en)
        fmap = torch.cat((en, dec), dim=1)  # F
        fmap = self.conv1(fmap)
        fmap = F.relu(fmap)
        if not self.mode == 'C':
            fmap_att = self.picanet(fmap)  # F_att
            x = torch.cat((fmap, fmap_att), 1)
            x = self.conv2(x)
            x = self.bn_feature(x)
            dec_out = F.relu(x)
            _y = self.conv3(dec_out)
            #_y = torch.sigmoid(_y)
        else:
            dec_out = self.conv2(fmap)
            _y = dec_out
            #_y = torch.sigmoid(dec_out)

        return dec_out, _y


class PicanetG(nn.Module):
    def __init__(self, size, in_channel):
        super(PicanetG, self).__init__()
        self.renet = Renet(size, in_channel, 100)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        #print(size)
        
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1)
        #print(x.size(), kernel.size())
        x = F.unfold(x, [5, 5], dilation=[2, 2])
        #print(x.size(), kernel.size())
        x = x.reshape(size[0], size[1], 100)
        kernel = kernel.reshape(size[0], 100, -1)
        #print(x.size(), kernel.size())
        x = torch.matmul(x, kernel)
        #print(x.size())
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x
'''
class PicanetG(nn.Module):
    def __init__(self, size, in_channel):
        super(PicanetG, self).__init__()
        self.renet = Renet(size, in_channel, 100)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1)
        x = F.unfold(x, [10, 10], dilation=[3, 3])
        x = x.reshape(size[0], size[1], 10 * 10)
        kernel = kernel.reshape(size[0], 100, -1)
        x = torch.matmul(x, kernel)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x
'''

class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        # print("Before unfold", x.shape)
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        # print("After unfold", x.shape)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        # print(x.shape, kernel.shape)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x


class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                                  bidirectional=True)  # each column
        self.conv = nn.Conv2d(512, out_channel, 1)

    def forward(self, *input):
        x = input[0]
        temp = []
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        for i in range(x.size()[2]):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(x.size()[1]):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        x = self.conv(x)
        return x
        
'''
class Unet(nn.Module):
    def __init__(self, cfg={'PicaNet': "GGLLL",
       'Size': [28, 28, 28, 56, 112, 224],
       'Channel': [1024, 512, 512, 256, 128, 64],
       'loss_ratio': [0.5, 0.5, 0.5, 0.8, 0.8, 1]}):
        super(Unet, self).__init__()
'''


class Decoder(nn.Module):
    def __init__(self, config, feat):
        super(Decoder, self).__init__()
        
        self.enc_conv = nn.Conv2d(feat[-1], feat[-1], 3, 1, 1)
        feat.append(feat[-1])
        
        module = config['module']
        self.decoder = nn.ModuleList()
        size = [10, 10, 20, 40, 80, 160]
        for i in range(5):
            assert module[i] == 'G' or module[i] == 'L'
            self.decoder.append(
                DecoderCell(size=size[i],
                            in_channel=feat[5 - i],
                            out_channel=feat[4 - i],
                            mode=module[i]))
        self.decoder.append(DecoderCell(size=size[5],
                                        in_channel=64,
                                        out_channel=1,
                                        mode='C'))
                                        
    def forward(self, en_out, x_size):
        en_out.append(self.enc_conv(en_out[-1]))
        dec = None
        pred = []
        for i in range(6):
            dec, _pred = self.decoder[i](en_out[5 - i], dec)
            pred.append(_pred)
            
        pred[-1] = F.interpolate(pred[-1], size=x_size, mode='bilinear', align_corners=True)
        
        out_dict = {}
        out_dict['sal'] = pred
        out_dict['final'] = pred[-1]
        return out_dict

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        
        self.encoder = encoder
        self.decoder = Decoder(config, feat)
        '''
        self.enc_conv = nn.Conv2d(feat[-1], feat[-1], 3, 1, 1)
        feat.append(feat[-1])
        
        module = config['module']
        self.decoder = nn.ModuleList()
        size = [10, 10, 20, 40, 80, 160]
        for i in range(5):
            assert module[i] == 'G' or module[i] == 'L'
            self.decoder.append(
                DecoderCell(size=size[i],
                            in_channel=feat[5 - i],
                            out_channel=feat[4 - i],
                            mode=module[i]))
        self.decoder.append(DecoderCell(size=size[5],
                                        in_channel=64,
                                        out_channel=1,
                                        mode='C'))
        '''

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        en_out = self.encoder(x)
        out_dict = self.decoder(en_out, x_size)
        '''
        en_out.append(self.enc_conv(en_out[-1]))
        #print(en_out[-1].size(), x.size())
        dec = None
        pred = []
        for i in range(6):
            dec, _pred = self.decoder[i](en_out[5 - i], dec)
            pred.append(_pred)
        pred[-1] = F.interpolate(pred[-1], size=x_size, mode='bilinear', align_corners=True)
        #print(pred[-1].size())
        out_dict = {}
        out_dict['sal'] = pred
        out_dict['final'] = pred[-1]
        '''
        return out_dict
