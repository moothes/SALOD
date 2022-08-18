import sys
import importlib
from data import Test_Dataset
from thop import profile
import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np

import argparse

from metric import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../dataset/', help='The name of network')
    parser.add_argument('--vals', default='all', help='Set the testing sets')
    parser.add_argument('--size', default=320, help='Set the testing sets')
    
    parser.add_argument('--pre_path', default='./maps', help='Weight path of network')
    
    params = parser.parse_args()
    config = vars(params)
    config['orig_size'] = True
    
    if config['vals'] == 'all':
        vals = ['PASCAL-S', 'ECSSD', 'HKU-IS', 'DUTS-TE', 'DUT-OMRON']
    else:
        vals = config['vals'].split(',')
        
    for val in vals:
        img_path = '{}/FS-SCFNet-R50/{}/'.format(config['pre_path'], val)
        if not os.path.exists(img_path):
            continue
        test_set = Test_Dataset(name=val, config=config)
        titer = test_set.size
        MR = MetricRecorder(titer)
        
        
        
        test_bar = Bar('Dataset {:10}:'.format(val), max=titer)
        for j in range(titer):
            _, gt, name = test_set.load_data(j)
            pred = Image.open(img_path + name + '.png').convert('L')
            out_shape = gt.shape
            
            pred = np.array(pred.resize((out_shape[::-1])))
            
            pred, gt = normalize_pil(pred, gt)
            MR.update(pre=pred, gt=gt)
            
                
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        #print('  MAE: {}, Max-F: {}, Maen-F: {}, SM: {}, EM: {}, Fbw: {}.'.format(mae, maxf, meanf, sm, em, wfm))
        print('  Max-F: {:.3f}, Maen-F: {:.3f}, Fbw: {:.3f}, MAE: {:.3f}, SM: {:.3f}, EM: {:.3f}.'.format(maxf, meanf, wfm, mae, sm, em))

    
if __name__ == "__main__":
    main()