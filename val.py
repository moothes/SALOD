import sys
import importlib
from data import Val_Dataset
from thop import profile
import torch
from torch.optim import SGD, Adam
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np

from base.framework_factory import load_framework

class PRF(nn.Module):
    def __init__(self, Y, step=255, stride=500):
        super(PRF, self).__init__()
        self.thresholds = range(step) #torch.linspace(0, end, steps=steps).cuda()
        self.Y = Y.squeeze().byte()
        
        self.step = step
        self.size = self.Y.size(0)
        self.stride = stride

    def forward(self, _Y):
        #print(self.Y.size(), _Y.size())
        _Y = _Y.squeeze() * 255
        TPs = torch.zeros((self.step, self.size)).cuda()
        T1s = torch.zeros((self.step, self.size)).cuda()
        T2 = torch.sum(self.Y, dim=(-2, -1)).float().unsqueeze(0)
        
        for threshold in self.thresholds:
            idx = 0
            while idx < self.size:
                end = min(self.size + 1, idx + self.stride)
                
                temp = _Y[idx:end] > threshold
                #print(temp.size())
                tp_temp = torch.sum(temp & self.Y[idx:end], dim=(-2, -1))
                #print(tp_temp)
                t1_temp = torch.sum(temp, dim=(-2, -1))
                #print(t1_temp)
                TPs[threshold, idx:end] = tp_temp
                T1s[threshold, idx:end] = t1_temp
                #print(tp_temp, t1_temp)
                idx += self.stride
            
        
        Ps = TPs / T1s
        #print(TPs)
        Rs = TPs / T2
        Ps[torch.isnan(Ps)] = 0
        Rs[torch.isnan(Rs)] = 0
        Ps = torch.mean(Ps, dim=-1)
        Rs = torch.mean(Rs, dim=-1)
        Fs = 1.3 * Ps * Rs / (Rs + 0.3 * Ps + 1e-9)
        

        return {'P':Ps, 'R':Rs, 'F':Fs}



def compute_mae(preds, labels):
    return np.mean(np.abs(preds - labels))

def val_model(model, val_sets, config, epoch=None):
    model.eval()
    if epoch is not None:
        weight_path = os.path.join(config['weight_path'], '{}_{}_{}.pth'.format(config['model_name'], config['sub'], epoch))
        if epoch >= (config['epoch'] - 5):
            torch.save(model.state_dict(), weight_path)
            
        
    st = time.time()
    for set_name, val_set in val_sets.items():
        #device = config['gpus'][-1]
        save_folder = os.path.join(config['save_path'], set_name)
        check_path(save_folder)
        
        image, gts, name = val_set.load_all_data()
        pred_maps = torch.zeros((val_set.size, 1, config['size'], config['size']))
        image = torch.tensor(image).float().cuda()
        gts = torch.tensor(gts)
        
        idx = 0
        stride = 1
        while idx < val_set.size:
            end = min(idx + stride, val_set.size + 1)
            #bimg = image[idx:end]
            #print(bimg.size())
            Y = model(image[idx:end])
            pred = Y['final'].sigmoid_().cpu().data
            #print(torch.max(pred))
            pred = pred / (torch.max(pred) + 1e-8)
            pred_maps[idx:end] = pred
            
            idx += stride
        
        image = 0
        model = 0
            
        pred_maps = pred_maps.squeeze(1)
        mae = compute_mae(pred_maps.numpy(), gts.numpy())
        
        '''
        #print(gts.size(), pred_maps.size())
        P = np.zeros((val_set.size, 255))
        R = np.zeros((val_set.size, 255))
        for i in range(pred_maps.size(0)):
            p, r = compute_pre_rec(gts[i].numpy() * 255, pred_maps[i].numpy() * 255)
            P[i] = p
            R[i] = r
        #print('original:')
        p = np.mean(P, axis=0)
        r = np.mean(R, axis=0)
        
        FM = (1 + 0.3) * p * r / (0.3 * p + r + 1e-8)
        print(np.mean(FM), np.max(FM))
        '''
        
        torch.cuda.empty_cache()
        prf = PRF(gts.cuda()).cuda()
        res = prf(pred_maps.cuda())
        #print(res['F'])
        Fs = res['F'].data.cpu().numpy()
        #print(np.mean(Fs), np.max(Fs))
        #print(Fs)
        #Fs = [F.cpu().data.numpy() for F in Fs]

        #prf.to(torch.device('cpu'))
        print('Dataset {:10}: MAE: {:.3f}, Max-F: {:.3f}, Mean-F: {:.3f}'.format(set_name, np.mean(mae), np.max(Fs), np.mean(Fs)))
        #print('Dataset {:10}: MAE: {:.3f}, Max-F: {:.3f}, Mean-F: {:.3f}.'.format(set_name, mae, max(Fs), np.mean(Fs)))
        
    print('Test using time: {}.'.format(round(time.time() - st, 3)))

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, _, _, _, saver = load_framework(net_name)
    
    saved_model = torch.load(config['weight'], map_location='cpu')
    model.load_state_dict(saved_model)
    
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Val_Dataset(name=set_name, config=config)
    
    #if not config['cpu']:
    model = model.cuda()
    
    val_model(model, test_sets, config)
        
if __name__ == "__main__":
    main()