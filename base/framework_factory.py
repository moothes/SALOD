import os
import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import importlib
import scipy
import scipy.ndimage
from torch.optim import SGD, Adam
import torch.optim.lr_scheduler as sche
import torch
import math

def load_framework(net_name):
    # Load Configure
    config = importlib.import_module('methods.{}.config'.format(net_name)).get_config()
    print(config)
    
    # Constructing network
    model = importlib.import_module('base.model').Network(net_name, config)
    #input = torch.randn(1, 3, 320, 320)
    #flops, params = profile(model, inputs=(input, ))
    #print('Network FLOPs: {} G, Params: {} M'.format(round(flops / 1e9, 2), round(params / 1e6, 2)))
    
    if config['loss'] == '':
        loss = importlib.import_module('methods.{}.loss'.format(net_name)).Loss
    else:
        loss = importlib.import_module('base.loss').Loss_factory(config)
    
    if config['save'] and os.path.exists('methods.{}.saver'.format(net_name)):
        saver = importlib.import_module('methods.{}.saver'.format(net_name)).Saver
    else:
        saver = None
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    
    if len(config['gpus']) > 1:
        model = torch.nn.DataParallel(model, device_ids=config['gpus']).module
    elif len(config['gpus']) == 1:
        model = model.cuda()
    
    # Set optimizer and schedule
    optim = config['optim']
    if optim == 'SGD':
        if 'params' in config.keys():
            module_lr = [{'params' : getattr(model.model, p[0]).parameters(), 'lr' : p[1]} for p in config['params']]
            optimizer = SGD(params=module_lr, lr=config['lr'], momentum=0.9, weight_decay=0.0005)
        else:
            optimizer = SGD(params=model.model.parameters(), lr=config['lr'], momentum=0.9)
    elif optim == 'Adam':
        optimizer = Adam(filter(lambda p: p.requires_grad, model.model.parameters()), lr=config['lr'], weight_decay=0.0005)
    
    schedule = config['schedule']
    if schedule == 'StepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['step_size'], gamma=config['gamma'])
    elif schedule == 'poly':
        scheduler = poly_scheduler(optimizer, config['epoch'], config['lr_decay'])
    elif schedule == 'pfa':
        scheduler = pfa_scheduler(optimizer, config['epoch'])
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.5)
    
    return config, model, optimizer, scheduler, loss, saver

def poly_scheduler(optimizer, total_num=50, lr_decay=0.9):
    def get_lr_coefficient(curr_epoch):
        nonlocal total_num
        coefficient = pow((1 - float(curr_epoch) / total_num), lr_decay)
        return coefficient

    scheduler = sche.LambdaLR(optimizer, lr_lambda=get_lr_coefficient)
    return scheduler

def pfa_scheduler(optimizer, total_num=50):
    def get_lr_coefficient(curr_epoch):
        nonlocal total_num
        e_drop = total_num / 8.
        coefficient = 1-abs(curr_epoch/total_num*2-1)
        return coefficient

    scheduler = sche.LambdaLR(optimizer, lr_lambda=get_lr_coefficient)
    return scheduler