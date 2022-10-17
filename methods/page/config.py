import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    strategy = 'base_sgd'
    parser = base_config(strategy)
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    config['clip_gradient'] = 5
    if config['loss'] == '':
        config['loss'] = {'sal': ['bi', 1, 1]}
    
    return config