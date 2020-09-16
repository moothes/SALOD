import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision

feat_dict = {'vgg': [64, 128, 256, 512, 512], 'resnet': [64, 256, 512, 1024, 2048]}

class RFC(nn.Module):
    def __init__(self, feats, resolution=0):
        super().__init__()
        
        if resolution == 0:
            raise ValueError('resolution should be in [16, 32, 64, 128, 256]')
        elif resolution == 16:
            self.layer1 = nn.Conv2d(feats[0], 64, 16, stride=16)
            self.layer2 = nn.Conv2d(feats[1], 64, 8, stride=8)
            self.layer3 = nn.Conv2d(feats[2], 64, 4, stride=4)
            self.layer4 = nn.Conv2d(feats[3], 64, 2, stride=2)
            self.layer5 = nn.Conv2d(feats[4], 64, 1)
            self.upsample = nn.ConvTranspose2d(64, 64, 16, stride=16)

        elif resolution == 32:
            self.layer1 = nn.Conv2d(feats[0], 64, 8, stride=8)
            self.layer2 = nn.Conv2d(feats[1], 64, 4, stride=4)
            self.layer3 = nn.Conv2d(feats[2], 64, 2, stride=2)
            self.layer4 = nn.Conv2d(feats[3], 64, 1)
            self.layer5 = nn.ConvTranspose2d(feats[4], 64, 2, stride=2)
            self.upsample = nn.ConvTranspose2d(64, 64, 8, stride=8)

        elif resolution == 64:
            self.layer1 = nn.Conv2d(feats[0], 64, 4, stride=4)
            self.layer2 = nn.Conv2d(feats[1], 64, 2, stride=2)
            self.layer3 = nn.Conv2d(feats[2], 64, 1)
            self.layer4 = nn.ConvTranspose2d(feats[3], 64, 2, stride=2)
            self.layer5 = nn.ConvTranspose2d(feats[4], 64, 4, stride=4)
            self.upsample = nn.ConvTranspose2d(64, 64, 4, stride=4)

        elif resolution == 128:
            self.layer1 = nn.Conv2d(feats[0], 64, 2, stride=2)
            self.layer2 = nn.Conv2d(feats[1], 64, 1)
            self.layer3 = nn.ConvTranspose2d(feats[2], 64, 2, stride=2)
            self.layer4 = nn.ConvTranspose2d(feats[3], 64, 4, stride=4)
            self.layer5 = nn.ConvTranspose2d(feats[4], 64, 8, stride=8)
            self.upsample = nn.ConvTranspose2d(64, 64, 2, stride=2)

        elif resolution == 256:
            self.layer1 = nn.Conv2d(feats[0], 64, 1)
            self.layer2 = nn.ConvTranspose2d(feats[1], 64, 2, stride=2)
            self.layer3 = nn.ConvTranspose2d(feats[2], 64, 4, stride=4)
            self.layer4 = nn.ConvTranspose2d(feats[3], 64, 8, stride=8)
            self.layer5 = nn.ConvTranspose2d(feats[4], 64, 16, stride=16)
            self.upsample = nn.ConvTranspose2d(64, 64, 1, stride=1)

        else:
            raise ValueError('resolution should be in [16, 32, 64, 128, 256]')
        self.conv_out = nn.Conv2d(320, 64, 1)
        self.bn = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, feature1, feature2, feature3, feature4, feature5):
        feature1 = self.layer1(feature1)
        feature2 = self.layer2(feature2)
        feature3 = self.layer3(feature3)
        feature4 = self.layer4(feature4)
        feature5 = self.layer5(feature5)
        
        feature = torch.cat([feature1, feature2, feature3, feature4, feature5], 1)
        feature = self.conv_out(feature)
        feature = self.bn(feature)
        feature = self.relu(feature)
        feature = self.upsample(feature)
        # Upsample(deconvolution) layer was placed after RFC block in original paper, but for convenience
        # I just put it here.
        return feature


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super().__init__()
        
        self.config = config
        self.encoder = encoder

        self.rfc1 = RFC(feat, 256)
        self.rfc2 = RFC(feat, 128)
        self.rfc3 = RFC(feat, 64)
        self.rfc4 = RFC(feat, 32)
        self.rfc5 = RFC(feat, 16)

        self.catconv1 = nn.Conv2d(64, 1, 3, padding=1)
        self.catconv2 = nn.Conv2d(65, 1, 3, padding=1)
        self.catconv3 = nn.Conv2d(65, 1, 3, padding=1)
        self.catconv4 = nn.Conv2d(65, 1, 3, padding=1)
        self.catdeconv = nn.ConvTranspose2d(65, 1, 3, padding=1)

        self.out_conv = nn.Sequential(
            nn.Conv2d(5, 64, 1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1)
        )
        #self.vgg16_weight_init()
        # print('model initialization complete')

    def forward(self, imgs, phase='test'):
        features = self.encoder(imgs)
        
        rfc1 = self.rfc1(*features)
        rfc2 = self.rfc2(*features)
        rfc3 = self.rfc3(*features)
        rfc4 = self.rfc4(*features)
        rfc5 = self.rfc5(*features)
        rfc5 = self.catconv1(rfc5)
        output1 = rfc5
        cat1 = torch.cat([rfc5, rfc4], 1)
        cat1 = self.catconv2(cat1)
        output2 = cat1
        cat2 = torch.cat([cat1, rfc3], 1)
        cat2 = self.catconv3(cat2)
        output3 = cat2
        cat3 = torch.cat([cat2, rfc2], 1)
        cat3 = self.catconv4(cat3)
        output4 = cat3
        cat4 = torch.cat([cat3, rfc1], 1)
        cat4 = self.catdeconv(cat4)
        output5 = cat4
        cat = torch.cat([output1, output2, output3, output4, output5], 1)
        output = self.out_conv(cat)
        
        if self.config['backbone'] == 'resnet':
            output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear')
            output1 = nn.functional.interpolate(output1, scale_factor=2, mode='bilinear')
            output2 = nn.functional.interpolate(output2, scale_factor=2, mode='bilinear')
            output3 = nn.functional.interpolate(output3, scale_factor=2, mode='bilinear')
            output4 = nn.functional.interpolate(output4, scale_factor=2, mode='bilinear')
            output5 = nn.functional.interpolate(output5, scale_factor=2, mode='bilinear')
        
        out_dict = {}
        out_dict['final'] = output
        out_dict['sal'] = [output1, output2, output3, output4, output5]
        #out_dict['final'] = self.get_pred(output)
        #out_dict['sal_pred'] = [self.get_pred(output) for output in [output1, output2, output3, output4, output5]]
        #[self.get_pred(output1), self.get_pred(output2), self.get_pred(output3), self.get_pred(output4), self.get_pred(output5)]

        return out_dict
        
    def get_pred(self, pred):
        pred = pred.exp()
        pred = pred[:, 0] / (pred[:, 0] + pred[:, 1])
        return pred.unsqueeze(1)
        