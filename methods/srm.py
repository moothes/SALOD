import timm
import torch
from torch import nn
from torch.nn import functional as F



custom_config = {'base'      : {'strategy': 'adam_base',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }
         
class Stage1(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv6 = nn.Conv2d(2048, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 2, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class Stage2(nn.Module):

    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.ppm = PPM()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.ppm(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

class PPM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block1 = nn.Sequential(  # 1*1 bin
            #nn.AvgPool2d(20, stride=20),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, 512, 1),
        )

        self.block2 = nn.Sequential(  # 2*2 bins
            #nn.AvgPool2d(10, stride=10),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(channel, 512, 1)
        )

        self.block3 = nn.Sequential(  # 3*3 bins
            #nn.AvgPool2d(8, stride=8),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(channel, 512, 1)
        )

        self.block4 = nn.Sequential(  # 6*6 bins
            #nn.AvgPool2d(4, stride=4),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(channel, 512, 1)
        )

    def forward(self, x):  # x shape: 24*24
        x_size = x.size()[2:]
        x1 = self.block1(x)
        x1 = nn.functional.interpolate(x1, size=x_size, mode='bilinear', align_corners=True)
        x2 = self.block2(x)
        x2 = nn.functional.interpolate(x2, size=x_size, mode='bilinear', align_corners=True)
        x3 = self.block3(x)
        x3 = nn.functional.interpolate(x3, size=x_size, mode='bilinear', align_corners=True)
        x4 = self.block4(x)
        x4 = nn.functional.interpolate(x4, size=x_size, mode='bilinear', align_corners=True)
        output = torch.cat([x, x1, x2, x3, x4], 1)
        return output


class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        
        self.config = config
        self.encoder = encoder
        self.conv6 = nn.Conv2d(feat[-1], 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 1, 3, padding=1)
        
        self.stage2 = timm.create_model(config['backbone'], features_only=True, pretrained=True)
        #if config['backbone'] == 'resnet50':
        #    self.stage2 = resnet50(True)        
        #elif config['backbone'] == 'vgg':
        #    self.stage2 = vgg(True)
        self.ppm = PPM(feat[-2])
        self.conv1 = nn.Conv2d(feat[-2]+2048, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, 3, padding=1)
        self.conv61 = nn.Conv2d(65, 256, 3, padding=1)
        self.conv71 = nn.Conv2d(256, 1, 3, padding=1)

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        _, _, _, _, feature1 = self.encoder(x)
        feature1 = self.conv7(F.relu(self.conv6(feature1)))
        output1 = F.interpolate(feature1, size=x_size, mode='bilinear', align_corners=True)
        #if self.config['backbone'] == 'resnet':
        feature1 = F.interpolate(feature1, scale_factor=2, mode='bilinear', align_corners=True)
        
        _, _, _, feature2, _ = self.stage2(x)
        #print(feature2.shape)
        feature2 = self.ppm(feature2)
        feature2 = self.conv1(feature2)
        feature2 = self.conv2(feature2)
        feature = torch.cat([feature1, feature2], 1)
        feature = self.conv61(feature)
        feature = self.conv71(feature)
        output2 = nn.functional.interpolate(feature, size=x_size, mode='bilinear', align_corners=True)
        
        out_dict = {}
        out_dict['final'] = output2
        out_dict['sal'] = [output1, output2]

        return out_dict
        
