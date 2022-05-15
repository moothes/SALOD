import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    strategy = 'base_adam'
    parser = base_config(strategy)
    # test
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    return config