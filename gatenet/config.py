import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    strategy = 'base_sgd'
    parser = base_config(strategy)
    # Add custom params here
    # parser.add_argument('--size', default=320, type=int, help='Input size')
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    #config['params'] = [['encoder', config['lr']*0.1]]
    config['ave_batch'] = 4
    # Config post-process
    if config['loss'] == '':
        config['loss'] = {'sal': ['bi', 1, 1]}
    
    return config