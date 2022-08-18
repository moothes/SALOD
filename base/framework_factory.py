import os
import torch
import numpy as np
import importlib
from torch.optim import SGD, Adam, AdamW
import torch.optim.lr_scheduler as sche

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def load_framework(net_name):
    # Load Configure
    config = importlib.import_module('methods.{}.config'.format(net_name)).get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    
    # Constructing network
    model = importlib.import_module('base.model').Network(net_name, config)
    
    if config['show_param']:
        from thop import profile
        input = torch.randn(1, 3, config['size'], config['size'])
        macs, params = profile(model, inputs=(input, ))
        params = params_count(model)
        print('MACs: {:.2f} G, Params: {:.2f} M.'.format(macs / 1e9, params / 1e6))
        
    if isinstance(config['loss'], str):
        if config['loss'] == '':
            config['loss'] = 'bi'
            config['lw'] = [1, 1]
        config['loss'] = {'sal': [config['loss'], *config['lw']]}
    loss = importlib.import_module('base.loss').Loss_factory(config['loss'])
        
    
    # Loading Saver if it exists
    if config['save'] and os.path.exists('methods/{}/saver.py'.format(net_name)):
        saver = importlib.import_module('methods.{}.saver'.format(net_name)).Saver
    else:
        saver = None
    
    gpus = range(len(config['gpus'].split(',')))
    
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus).module
    model = model.cuda()
    
    # Set optimizer and schedule
    optimizer, schedule = importlib.import_module('base.strategy').Strategy(model, config)
    
    return config, model, optimizer, schedule, loss, saver
