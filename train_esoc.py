import sys
import os
import time

#from framework_factory import load_framework
#from thop import profile
from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data_esoc import *
from test import test_model
from val import val_model
import torch
from torch.nn import utils
from framework_factory import load_framework


def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    #print(model)
    
    # Loading datasets
    train_loader = get_loader(name='ESOC', config=config)
    test_sets = OrderedDict()
    test_sets['ESOC'] = ESOC_Test(name='ESOC', config=config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    ave_batch = config['ave_batch']
    
    #print(train_set.size)
    num_iter = len(train_loader) # // config['batch'] 
    batch_idx = 0
    model.zero_grad()

    for epoch in range(1, config['epoch'] + 1):
        model.train()
        torch.cuda.empty_cache()
        #model.eval()
        sche.step()
        
        if debug:
            #model.eval()
            test_model(model, test_sets, config, epoch)
            #val_model(model, test_sets, config, epoch)
            
        print('---------------------------------------------------------------------------')
        bar = Bar('{:10}-{:8} | epoch {}:'.format(net_name, config['sub'], epoch), max=num_iter)

        st = time.time()
        #loss_batch = 0
        loss_count = 0
        optim.zero_grad()
        for i, pack in enumerate(train_loader, start=1):
            images, gts = pack
            
            images, gts= images.float().cuda(), gts.float().cuda()
            
            if not config['orig_size']:
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
            
            if i % ave_batch == 0:                
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.model.parameters(), config['clip_gradient'])
                optim.step()   
                optim.zero_grad()
                
                batch_idx = 0
            Bar.suffix = '{}/{} | loss: {}'.format(i, num_iter, loss_count / i)
            bar.next()

        bar.finish()
        print('epoch: {},  time: {:.2f}s, loss: {:.5f}.'.format(epoch, time.time() - st, loss_count / num_iter))
        
        #val_model(model, test_sets, config, epoch)
    test_model(model, test_sets, config, epoch)

        
        
if __name__ == "__main__":
    main()