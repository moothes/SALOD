import sys
import importlib
from data import Test_Dataset
#from data_esod import ESOD_Test
import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np

from base.framework_factory import load_framework
from metric import *
#from framework_factory import load_framework

def eval(pred, gt):
    mae = eval_mae(pred, gt)
    fm = eval_f(pred * 255, gt * 255)
    em = eval_e(pred, gt)
    sm = eval_s(pred, gt)
    fbw = eval_fbw(pred, gt)
    
    return mae, fm, em, sm, fbw

def test_model(model, test_sets, config, epoch=None, saver=None):
    model.eval()
    if epoch is not None:
        weight_path = os.path.join(config['weight_path'], '{}_{}_{}.pth'.format(config['model_name'], config['sub'], epoch))
        torch.save(model.state_dict(), weight_path)
    
    st = time.time()
    for set_name, test_set in test_sets.items():
        save_folder = os.path.join(config['save_path'], set_name)
        check_path(save_folder)
        
        titer = test_set.size
        MR = MetricRecorder(titer)
        
        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        for j in range(titer):
            image, gt, name = test_set.load_data(j)
            Y = model(image.cuda())
            pred = Y['final'][0, 0].sigmoid_().cpu().data.numpy()
            out_shape = gt.shape
            
            pred = np.array(Image.fromarray(pred).resize((out_shape[::-1])))
            
            pred = np.round(pred * 255) / 255.
            MR.update(pre=pred, gt=gt)
            
            # save predictions
            if config['save']:
                fnl_folder = os.path.join(save_folder, 'final')
                check_path(fnl_folder)
                im_path = os.path.join(fnl_folder, name + '.png')
                Image.fromarray((pred * 255)).convert('L').save(im_path)
                
                if saver is not None:
                    saver(Y, gt, name, save_folder, config)
                    pass
                
            Bar.suffix = '{}/{}'.format(j, titer)
            test_bar.next()
        
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        #print('  MAE: {}, Max-F: {}, Maen-F: {}, SM: {}, EM: {}, Fbw: {}.'.format(mae, maxf, meanf, sm, em, wfm))
        print('  Max-F: {}, Maen-F: {}, Fbw: {}, MAE: {}, SM: {}, EM: {}.'.format(maxf, meanf, wfm, mae, sm, em))
        
    print('Test using time: {}.'.format(round(time.time() - st, 3)))

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, _, _, _, saver = load_framework(net_name)
    
    #model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
    saved_model = torch.load(config['weight'], map_location='cpu')
    new_name = {}
    for k, v in saved_model.items():
        if k.startswith('model'):
            new_name[k[6:]] = v
        else:
            new_name[k] = v
    model.load_state_dict(new_name)

    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    model = model.cuda()
    
    test_model(model, test_sets, config, saver=saver)
        
if __name__ == "__main__":
    main()
