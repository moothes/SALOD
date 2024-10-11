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
         

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()

        self.encoder = encoder
        #self.decoder = nn.Conv2d(feat[-1], 1, 1)
    
    def forward(self, x, phase='test'):
        # phase: enable different operations in training/testing phases. Useful in several networks.
        x_size = x.size()[2:]
        enc_feats = self.encoder(x)
        #x = self.decoder(enc_feats[-1])
        pred = torch.mean(enc_feats[-1], dim=1, keepdim=True)
        pred = nn.functional.interpolate(pred, size=x_size, mode='bilinear')
        
        # All output are save in a Dict, while the key of final prediction is 'final'
        OutDict = {}
        OutDict['sal'] = [pred, ]
        OutDict['final'] = pred
        return OutDict
