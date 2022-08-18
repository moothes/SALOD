import sys
import argparse  
import os
from base.config import base_config, cfg_convert
from util import check_path


def get_config():
    strategy = 'sche_f3net' #'base_adam'
    
    parser = base_config(strategy)
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} network with {} backbone using Gpu: {}'.format(config['model_name'], config['backbone'], config['gpus']))
    
    config['ave_batch'] = 2
    if config['loss'] == '':
        config['loss'] = {'sal': ['bi', 1, 1]}
    
    return config