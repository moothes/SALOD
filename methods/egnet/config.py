import sys
import argparse  
import os
from base.config import base_config, cfg_convert
from util import *


def get_config():
    cfg_dict = {
        'optim': 'Adam',
        'schedule': 'StepLR',
        'lr': 5e-5,
        'batch': 4,
        'ave_batch': 2,
        'epoch': 30,
        'step_size': '20',
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
        config['optim'] = 'Adam'
    if config['schedule'] == '':
        config['schedule'] = 'StepLR'
    if config['lr'] == 0:
        config['lr'] = 5e-5
    if config['batch'] == 0:
        config['batch'] = 4
    if config['ave_batch'] == 0:
        config['ave_batch'] = 2
    if config['epoch'] == 0:
        config['epoch'] = 30
    if config['step_size'] == '':
        config['step_size'] = '20'
    if config['gamma'] == 0:
        config['gamma'] = 0.1
    if config['size'] == 0:
        config['size'] = 320
    '''
    return config