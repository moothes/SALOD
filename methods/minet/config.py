import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    cfg_dict = {
        'optim': 'SGD',
        'schedule': 'StepLR', #'poly'
        'lr': 0.001,
        'batch': 8, # 4
        'ave_batch': 1,
        'epoch': 30, # 50
        'step_size': '20', #''
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }
    
    parser = base_config(cfg_dict)
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    '''
    if config['optim'] == '':
        config['optim'] = 'SGD'
    if config['schedule'] == '':
        config['schedule'] = 'poly'
    if config['lr'] == 0:
        config['lr'] = 0.001
    if config['batch'] == 0:
        config['batch'] = 4
    if config['ave_batch'] == 0:
        config['ave_batch'] = 1
    if config['epoch'] == 0:
        config['epoch'] = 50
    if config['gamma'] == 0:
        config['gamma'] = 0.1
    if config['size'] == 0:
        config['size'] = 320
    '''
    config['lr_decay'] = 0.9
    
    return config