import sys
import argparse  
import os
from base.config import base_config, cfg_convert


def get_config():
    strategy = 'sche_f3net' # 'sche_test'
    parser = base_config(strategy)
    
    params = parser.parse_args()
    config = vars(params)
    cfg_convert(config)
    print('Training {} with {} backbone using {} on GPU: {}'.format(config['model_name'], config['backbone'], strategy, config['gpus']))
    
    config['ave_batch'] = 2
    
    if config['loss'] == '':
        config['loss'] = {'sal': ['bi', 1, 1]}
    
    return config