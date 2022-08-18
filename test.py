import sys
import importlib
import torch
import time
import os
import cv2
from PIL import Image
import numpy as np
from collections import OrderedDict
from progress.bar import Bar

from util import *
from metric import *
from data import Test_Dataset
from base.framework_factory import load_framework   

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
        scores = []
        
        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        for j in range(titer):
            image, gt, name = test_set.load_data(j)
            Y = model(image.cuda())
            pred = Y['final'][0, 0].sigmoid_().cpu().data.numpy()
            
            out_shape = gt.shape
            
            #pred = np.array(Image.fromarray(pred).resize((out_shape[::-1]), resample=0))
            pred = cv2.resize(pred, (out_shape[::-1]), interpolation=cv2.INTER_LINEAR)
            
            pred, gt = normalize_pil(pred, gt)
            pred = np.clip(np.round(pred * 255) / 255., 0, 1)
            MR.update(pre=pred, gt=gt)
            
            #scores.append(get_scores(pred, gt))
            #print(get_scores(pred, gt))
            
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
        
        #scores = np.array(scores)
        #print(np.mean(scores, axis=0))
        
        mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
        #print('  MAE: {}, Max-F: {}, Maen-F: {}, SM: {}, EM: {}, Fbw: {}.'.format(mae, maxf, meanf, sm, em, wfm))
        print('  Max-F: {:.3f}, Maen-F: {:.3f}, Fbw: {:.3f}, MAE: {:.3f}, SM: {:.3f}, EM: {:.3f}.'.format(maxf, meanf, wfm, mae, sm, em))
        
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