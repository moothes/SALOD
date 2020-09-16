import sys
import os
import time
import random

#from thop import profile
from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data import *
from test import test_model
from val import val_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework


def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    
    # Loading datasets
    train_loader = get_loader(name=config['trset'], config=config)
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Val_Dataset(name=set_name, config=config)
        #test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    ave_batch = config['ave_batch']
    batch_idx = 0
    
    for epoch in range(1, config['epoch'] + 1):
        sche.step()
        
        if debug:
            #test_model(model, test_sets, config, epoch)
            val_model(model, test_sets, config, epoch)
            
        print('---------------------------------------------------------------------------')
        bar = Bar('{:10}-{:8} | epoch {}:'.format(net_name, config['sub'], epoch), max=num_iter)

        model.train()
        st = time.time()
        loss_count = 0
        optim.zero_grad()
        for i, pack in enumerate(train_loader, start=1):
            images, gts = pack
            images, gts= images.float().cuda(), gts.float().cuda()
            
            if config['multi']:
                #print('----------multi-scale training------------')
                if net_name == 'picanet':
                    # picanet only support 320*320 input now!
                    # picanet doesn't support multi-scale training, so we crop images to same sizes to simulate it.
                    input_size = config['size']
                    images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
                    
                    scales = [32, 16, 0] 
                    scale = np.random.choice(scales, 1)
                    w_start = int(random.random() * scale)
                    h_start = int(random.random() * scale)
                    new_size = int(input_size - scale)
                    images = images[:, :, h_start:h_start+new_size, w_start:w_start+new_size]
                    gts = gts[:, :, h_start:h_start+new_size, w_start:w_start+new_size]
                    
                    images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
                else:
                    scales = [-1, 0, 1] 
                    input_size = config['size']
                    input_size += int(np.random.choice(scales, 1) * 64)
                    images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
            
            Y = model(images, 'train')
            loss = model_loss(Y, gts, config) / ave_batch
            loss_count += loss.data
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            Bar.suffix = '{}/{} | loss: {}'.format(i, num_iter, loss_count / i)
            bar.next()

        bar.finish()
        print('epoch: {},  time: {:.2f}s, loss: {:.5f}.'.format(epoch, time.time() - st, loss_count / num_iter))
        
        #test_model(model, test_sets, config, epoch)
        val_model(model, test_sets, config, epoch)

if __name__ == "__main__":
    main()