import sys
import argparse  
import os
from util import *

def base_config(strategy):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', default='', help='Model name')
    parser.add_argument('--backbone', default='resnet50', help='Set the backbone of the model')
    parser.add_argument('--show_param', action='store_true') # show the number of parameter
    
    # Training schedule
    parser.add_argument('--sub', default='base', help='Job name')
    parser.add_argument('--strategy', default=strategy, help='Training strategy, see base/strategy.py')
    parser.add_argument('--clip_gradient', default=0, type=float, help='Max gradient')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--data_aug', action='store_false', help='Data augmentation, only random crop')
    parser.add_argument('--multi', action='store_false', help='Multi-scale training')
    parser.add_argument('--gpus', default='0', type=str, help='Set the gpu devices')
    
    # Data setting
    parser.add_argument('--size', default=320, type=int, help='Input size')
    parser.add_argument('--trset', default='SALOD', help='Set the traing set') # DUTS-TR, COD-TR, SALOD, simple
    parser.add_argument('--vals', default='all', help='Set the testing sets')
    parser.add_argument('--data_path', default='../dataset', help='Dataset path')
    parser.add_argument('--save_path', default='./result/', help='Save path')
    parser.add_argument('--weight_path', default='./weight/', help='Weight path')
    
    parser.add_argument('--debug', action='store_true') # Test model before training

    # Testing
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--weight', default='', help='Loading weight file')
    parser.add_argument('--save', action='store_true', help='Whether save result')
    parser.add_argument('--test_batch', default=1, type=int, help='Batch Size for Testing')
    
    # Use for SALOD dataset
    parser.add_argument('--train_split', default=10000, type=int, help='Use for SALOD dataset')
    
    # Construct loss by loss_factory. More details in base/loss.py.
    parser.add_argument('--loss', default='', type=str, help='Losses for networks')
    parser.add_argument('--lw', default='', type=str, help='Weights for losses')
    
    return parser
    
def cfg_convert(parser):
    params = parser.parse_args()
    config = vars(params)
    print(f"Training {config['model_name']} with {config['backbone']} backbone using {config['strategy']} strategy on GPU: {config['gpus']}")

    if config['trset'] == 'SALOD':
        config['vals'] = ['SALOD']
    elif config['trset'] == 'simple':
        config['vals'] = ['tough', 'normal']
    elif config['trset'] == 'DUTS-TR':
        if config['vals'] == 'all':
            config['vals'] = ['PASCAL-S', 'ECSSD', 'HKU-IS', 'DUTS-TE', 'DUT-OMRON']
        else:
            config['vals'] = config['vals'].split(',')
    elif config['trset'] == 'COD-TR':
        if config['vals'] == 'all':
            config['vals'] = ['COD-TE', 'NC4K', 'CAMO-TE']
        else:
            config['vals'] = config['vals'].split(',')
    else:
        config['vals'] = config['vals'].split(',')
    
    save_path = os.path.join(config['save_path'], config['model_name'], config['backbone'], config['sub'])
    check_path(save_path)
    config['save_path'] = save_path
    
    weight_path = os.path.join(config['weight_path'], config['model_name'], config['backbone'], config['sub'])
    check_path(weight_path)
    config['weight_path'] = weight_path
    
    return config