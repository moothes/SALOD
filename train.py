import sys
import os
import time
import random

#from thop import profile
from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data import get_loader, Test_Dataset
from test import test_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework

torch.set_printoptions(precision=5)

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    ave_batch = config['agg_batch'] // config['batch']
    # agg_batch: batch size for backwarding.
    # batch: batch size when loading to gpus. Decided by the GPU memory.
    print(sorted(config.items()))
    
    
    # Loading datasets
    train_loader = get_loader(config)
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    start_epoch = 1
    if config['resume']:
        saved_model = torch.load(config['weight'], map_location='cpu')
        if config['num_gpu'] > 1:
            model.module.load_state_dict(saved_model)
        else:
            model.load_state_dict(saved_model)
            
        start_epoch = int(config['weight'].split('_')[-1].split('.')[0]) + 1
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    #ave_batch = config['ave_batch']
    trset = config['trset']
    model.zero_grad()
    for epoch in range(start_epoch, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if debug:
            test_model(model, test_sets, config, epoch)
        
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=num_iter)
        
        config['cur_epoch'] = epoch
        config['iter_per_epoch'] = num_iter
        st = time.time()
        optim.zero_grad()
        loss_count = 0
        batch_idx = 0
        #sche.step()
        for i, pack in enumerate(train_loader, start=1):
            current_iter = (epoch - 1) * num_iter + i
            total_iter = num_epoch * num_iter
            #print('iter: ', total_iter, current_iter)
            
            sche(optim, current_iter, total_iter, config)
            
            
            images, gts = pack
            images, gts= images.float().cuda(), gts.float().cuda()
            
            if config['multi']:
                if net_name == 'picanet':
                    # picanet only support 320*320 input now!
                    # picanet doesn't support multi-scale training, so we crop images to same sizes to simulate it.
                    input_size = config['size']
                    images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
                    
                    scales = [16, 8, 0] 
                    scale = np.random.choice(scales, 1)
                    w_start = int(random.random() * scale)
                    h_start = int(random.random() * scale)
                    new_size = int(input_size - scale)
                    images = images[:, :, h_start:h_start+new_size, w_start:w_start+new_size]
                    gts = gts[:, :, h_start:h_start+new_size, w_start:w_start+new_size]
                    
                    images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
                else:
                    #scales = [-1, 0, 1] 
                    scales = [-2, -1, 0, 1, 2] 
                    input_size = config['size']
                    input_size += int(np.random.choice(scales, 1) * 64)
                    #input_size += int(np.random.choice(scales, 1) * 32)
                    images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
                    
            Y = model(images, 'train')
            loss = model_loss(Y, gts, config) / ave_batch
            loss_count += loss.data
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            lrs = ','.join([format(param['lr'], ".1e") for param in optim.param_groups])
            Bar.suffix = '{:4}/{:4} | loss: {:1.3f}, LRs: [{}], time: {:1.3f}.'.format(i, num_iter, float(loss_count / i), lrs, time.time() - st)
            bar.next()

        bar.finish()
        weight_path = os.path.join(config['weight_path'], '{}_{}_{}_{}.pth'.format(config['model_name'], config['backbone'], config['sub'], epoch))
        torch.save(model.state_dict(), weight_path)
        
        
        if trset in ('DUTS-TR', 'MSB-TR', 'COD-TR') and epoch > num_epoch - 10:
            test_model(model, test_sets, config, epoch)
        #test_model(model, test_sets, config, epoch)
            
    if trset != 'DUTS-TR':
        test_model(model, test_sets, config, epoch)

if __name__ == "__main__":
    main()