import sys
import argparse  
import os
from base.config import base_config, cfg_convert
from util import check_path


def get_config():
    strategy = 'base_sgd'
    parser = base_config(strategy)
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    config['ws'] = [1., 0.8, 0.6, 0.4]
    
    return config