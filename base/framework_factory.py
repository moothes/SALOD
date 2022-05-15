import os
import torch
import numpy as np
import importlib
from torch.optim import SGD, Adam, AdamW
import torch.optim.lr_scheduler as sche
#from fvcore.nn.flop_count import flop_count

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def load_framework(net_name):
    # Load Configure
    config = importlib.import_module('methods.{}.config'.format(net_name)).get_config()
    #print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    
    # Constructing network
    model = importlib.import_module('base.model').Network(net_name, config)
    
    if config['show_param']:
        from thop import profile
        input = torch.randn(1, 3, config['size'], config['size'])
        macs, params = profile(model, inputs=(input, ))
        params = params_count(model)
        print('MACs: {:.2f} G, Params: {:.2f} M.'.format(macs / 1e9, params / 1e6))
        
    if config['loss'] == '':
        loss = importlib.import_module('methods.{}.loss'.format(net_name)).Loss
    else:
        loss = importlib.import_module('base.loss').Loss_factory(config)
    
    # Loading Saver if it exists
    #print(os.path.exists('methods/{}/saver.py'.format(net_name)))
    if config['save'] and os.path.exists('methods/{}/saver.py'.format(net_name)):
        saver = importlib.import_module('methods.{}.saver'.format(net_name)).Saver
    else:
        saver = None
    #print(saver is None)
    
    gpus = range(len(config['gpus'].split(','))) # [int(x) for x in config['gpus'].split(',')]
    
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus).module
    model = model.cuda()
    
    optimizer, schedule = importlib.import_module('base.strategy').Strategy(model, config)
    # Set optimizer and schedule
    '''
    optim = config['optim']
    if optim == 'SGD':
        if 'params' in config.keys():
            module_lr = [{'params' : getattr(model, p[0]).parameters(), 'lr' : p[1]} for p in config['params']]
            optimizer = SGD(params=module_lr, lr=config['lr'], momentum=0.9, weight_decay=0.0005)
        else:
            encoder = []
            others = []
            for param in model.named_parameters():
                if 'encoder.' in param[0]:
                    encoder.append(param[1])
                else:
                    others.append(param[1])
                    
            module_lr = [{'params' : encoder, 'lr' : config['lr']*0.1}, {'params' : others, 'lr' : config['lr']}]
            optimizer = SGD(params=module_lr, lr=config['lr'], momentum=0.9)
    elif optim == 'Adam':
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'], weight_decay=0.0005)
    elif optim == 'AdamW':
        encoder = []
        others = []
        for param in model.named_parameters():
            if 'encoder.' in param[0]:
                encoder.append(param[1])
            else:
                others.append(param[1])
                
        module_lr = [{'params' : encoder, 'lr' : config['lr']*0.1}, {'params' : others, 'lr' : config['lr']}]
        optimizer = AdamW(module_lr, lr = config['lr'], weight_decay=5.e-2)  # 
        
    
    # If get_config function doesn't return a valid schedule, it will be set here.
    if schedule is None:
        schedule = config['schedule']
        if schedule == 'StepLR':
            scheduler = sche.MultiStepLR(optimizer, milestones=config['step_size'], gamma=config['gamma'])
        elif schedule == 'poly':
            scheduler = poly_scheduler(optimizer, config['epoch'], config['lr_decay'])
        elif schedule == 'pfa':
            scheduler = pfa_scheduler(optimizer, config['epoch'])
        else:
            scheduler = sche.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.5)
    '''
    
    return config, model, optimizer, schedule, loss, saver
