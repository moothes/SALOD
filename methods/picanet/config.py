import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    cfg_dict = {
        'optim': 'SGD',
        'schedule': 'StepLR',
        'lr': 0.01,
        'batch': 4,
        'ave_batch': 2,
        'epoch': 30,
        'step_size': '20',
        'gamma': 0.1,
        'clip_gradient': 0,
        'test_batch': 1,
    }
    
    parser = base_config(cfg_dict)
    # Add custom params here
    # parser.add_argument('--size', default=320, type=int, help='Input size')
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    config['params'] = [['encoder', config['lr'] / 10], ['decoder', config['lr']]]
    # Config post-process
    config['ws'] = [0.5, 0.5, 0.5, 0.8, 0.8, 1.]
    config['module'] = 'GGLLL'
    
    return config