import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    strategy = 'base_sgd'
    parser = base_config(strategy)
    
    parser.add_argument('--dropout', action='store_false')
    parser.add_argument('--with_CA', action='store_false')
    parser.add_argument('--with_SA', action='store_false')
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    
    return config