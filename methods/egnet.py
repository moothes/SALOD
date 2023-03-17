import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np


custom_config = {'base'      : {'strategy': 'base_adam',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }
         


config_vgg = {'convert': [[128,256,512,512,512],[64,128,256,512,512]], 'merge1': [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}  # no convert layer, no conv6
config_resnet = {'convert': [[64,256,512,1024,2048],[128,256,512,512,512]], 'deep_pool': [[512, 512, 256, 256, 128], [512, 256, 256, 128, 128], [False, True, True, True, False], [True, True, True, True, False]], 'score': 256, 'edgeinfo':[[16, 16, 16, 16], 128, [16,8,4,2]],'edgeinfoc':[64,128], 'block': [[512, [16]], [256, [16]], [256, [16]], [128, [16]]], 'fuse': [[16, 16, 16, 16], True], 'fuse_ratio': [[16,1], [8,1], [4,1], [2,1]],  'merge1': [[128, 256, 128, 3,1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],[512, 0, 512, 7, 3]], 'merge2': [[128], [256, 512, 512, 512]]}

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
          
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up0)

    def forward(self, list_x):
        resl = []
        #for i in range(len(list_x)):
        for i, x in enumerate(list_x):
            resl.append(self.convert0[i](x))
        return resl

class MergeLayer1(nn.Module): # list_k: [[64, 512, 64], [128, 512, 128], [256, 0, 256] ... ]
    def __init__(self, list_k):
        super(MergeLayer1, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for ik in list_k:
            if ik[1] > 0:
                trans.append(nn.Sequential(nn.Conv2d(ik[1], ik[0], 1, 1, bias=False), nn.ReLU(inplace=True)))

            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True), nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True)))
            score.append(nn.Conv2d(ik[2], 1, 3, 1, 1))
        trans.append(nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)
        self.relu =nn.ReLU()

    def forward(self, list_x, x_size):
        up_edge, up_sal, edge_feature, sal_feature = [], [], [], []
        
        num_f = len(list_x)
        tmp = self.up[num_f - 1](list_x[num_f-1])
        sal_feature.append(tmp)
        U_tmp = tmp
        up_sal.append(F.interpolate(self.score[num_f - 1](tmp), x_size, mode='bilinear', align_corners=True))
        
        for j in range(2, num_f ):
            i = num_f - j
             
            if list_x[i].size()[1] < U_tmp.size()[1]:
                U_tmp = list_x[i] + F.interpolate((self.trans[i](U_tmp)), list_x[i].size()[2:], mode='bilinear', align_corners=True)
            else:
                U_tmp = list_x[i] + F.interpolate((U_tmp), list_x[i].size()[2:], mode='bilinear', align_corners=True)
            
            tmp = self.up[i](U_tmp)
            U_tmp = tmp
            sal_feature.append(tmp)
            up_sal.append(F.interpolate(self.score[i](tmp), x_size, mode='bilinear', align_corners=True))

        U_tmp = list_x[0] + F.interpolate((self.trans[-1](sal_feature[0])), list_x[0].size()[2:], mode='bilinear', align_corners=True)
        tmp = self.up[0](U_tmp)
        edge_feature.append(tmp)
       
        up_edge.append(F.interpolate(self.score[0](tmp), x_size, mode='bilinear', align_corners=True)) 
        return up_edge, edge_feature, up_sal, sal_feature        
        
class MergeLayer2(nn.Module): 
    def __init__(self, list_k):
        super(MergeLayer2, self).__init__()
        self.list_k = list_k
        trans, up, score = [], [], []
        for i in list_k[0]:
            tmp = []
            tmp_up = []
            tmp_score = []
            feature_k = [[3,1],[5,2], [5,2], [7,3]]
            for idx, j in enumerate(list_k[1]):
                tmp.append(nn.Sequential(nn.Conv2d(j, i, 1, 1, bias=False), nn.ReLU(inplace=True)))

                tmp_up.append(nn.Sequential(nn.Conv2d(i , i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i,  feature_k[idx][0],1 , feature_k[idx][1]), nn.ReLU(inplace=True), nn.Conv2d(i, i, feature_k[idx][0], 1, feature_k[idx][1]), nn.ReLU(inplace=True)))
                tmp_score.append(nn.Conv2d(i, 1, 3, 1, 1))
            trans.append(nn.ModuleList(tmp))

            up.append(nn.ModuleList(tmp_up))
            score.append(nn.ModuleList(tmp_score))

        self.trans, self.up, self.score = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score)       
        self.final_score = nn.Sequential(nn.Conv2d(list_k[0][0], list_k[0][0], 5, 1, 2), nn.ReLU(inplace=True), nn.Conv2d(list_k[0][0], 1, 3, 1, 1))
        self.relu =nn.ReLU()

    def forward(self, list_x, list_y, x_size):
        up_score, tmp_feature = [], []
        list_y = list_y[::-1]

        
        for i, i_x in enumerate(list_x):
            for j, j_x in enumerate(list_y):                              
                tmp = F.interpolate(self.trans[i][j](j_x), i_x.size()[2:], mode='bilinear', align_corners=True) + i_x                
                tmp_f = self.up[i][j](tmp)             
                up_score.append(F.interpolate(self.score[i][j](tmp_f), x_size, mode='bilinear', align_corners=True))                  
                tmp_feature.append(tmp_f)
       
        tmp_fea = tmp_feature[0]
        for i_fea in range(len(tmp_feature) - 1):
            tmp_fea = self.relu(torch.add(tmp_fea, F.interpolate((tmp_feature[i_fea+1]), tmp_feature[0].size()[2:], mode='bilinear', align_corners=True)))
        up_score.append(F.interpolate(self.final_score(tmp_fea), x_size, mode='bilinear', align_corners=True))
      
        return up_score

# extra part
def extra_layer(base_model_cfg):
    if base_model_cfg == 'vgg':
        bb_config = config_vgg
    elif base_model_cfg == 'resnet50':
        bb_config = config_resnet
    else:
        print(base_model_cfg)
    merge1_layers = MergeLayer1(bb_config['merge1'])
    merge2_layers = MergeLayer2(bb_config['merge2'])

    return merge1_layers, merge2_layers

class decoder(nn.Module):
    def __init__(self, base_model_cfg, merge1_layers, merge2_layers):
        super(decoder, self).__init__()
        self.base_model_cfg = base_model_cfg
        if self.base_model_cfg == 'vgg':

            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

        elif self.base_model_cfg == 'resnet50':
            self.convert = ConvertLayer(config_resnet['convert'])
            self.merge1 = merge1_layers
            self.merge2 = merge2_layers

    def forward(self, conv2merge, x_size):
        #x_size = x.size()[2:]
        #conv2merge = self.encoder(x)        
        if self.base_model_cfg == 'resnet50':            
            conv2merge = self.convert(conv2merge)
        else: 
            conv2merge = conv2merge[1:]
        up_edge, edge_feature, up_sal, sal_feature = self.merge1(conv2merge, x_size)
        up_sal_final = self.merge2(edge_feature, sal_feature, x_size)
        return up_edge, up_sal, up_sal_final

# TUN network
#class TUN_bone(nn.Module):
class Network(nn.Module):
    #def __init__(self, base_model_cfg, base, merge1_layers, merge2_layers):
    def __init__(self, config, encoder, feat):
        super(Network, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder(config['backbone'], *extra_layer(config['backbone']))

    def forward(self, x, phase='test'):
        x_size = x.size()[2:]
        conv2merge = self.encoder(x)        
        up_edge, up_sal, up_sal_final = self.decoder(conv2merge, x_size)
        
        out_dict = {}
        out_dict['edge'] = up_edge
        out_dict['sal'] = up_sal + up_sal_final
        #out_dict['sal_f'] = up_sal_final
        out_dict['final'] = up_sal_final[-1]
        return out_dict
