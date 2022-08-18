import sys
import argparse  
import os
from util import *

def base_config(strategy):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', default='', help='Model name')
    parser.add_argument('--backbone', default='resnet', help='Set the backbone of the model')
    parser.add_argument('--gpus', default='0', type=str, help='Set the gpu devices')
    
    # Training schedule
    parser.add_argument('--strategy', default=strategy, help='Training strategy, see base/strategy.py')
    parser.add_argument('--ave_batch', default=1, type=int, help='Number of batches for each backpropagation')
    parser.add_argument('--clip_gradient', default=0, type=float, help='Max gradient')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--sub', default='base', help='Job name')
    parser.add_argument('--data_aug', action='store_false', help='Data augmentation, only random crop')
    parser.add_argument('--multi', action='store_false', help='Multi-scale training')
    
    # Data setting
    parser.add_argument('--size', default=320, type=int, help='Input size')
    parser.add_argument('--trset', default='DUTS-TR', help='Set the traing set') # DUTS-TR, COD-TR, SALOD, simple
    parser.add_argument('--vals', default='all', help='Set the testing sets')
    parser.add_argument('--data_path', default='../dataset', help='Dataset path')
    parser.add_argument('--save_path', default='./result/', help='Save path')
    parser.add_argument('--weight_path', default='./weight/', help='Weight path')
    parser.add_argument('--debug', action='store_true')

    # Testing
    parser.add_argument('--weight', default='', help='Loading weight file')
    parser.add_argument('--save', action='store_true', help='Whether save result')
    parser.add_argument('--test_batch', default=1, type=int, help='Batch Size for Testing')
    parser.add_argument('--show_param', action='store_true')
    
    # For SALOD dataset
    parser.add_argument('--train_split', default=10000, type=int, help='Use for SALOD dataset')
    
    # Construct loss by loss_factory. More details in base/loss.py.
    parser.add_argument('--loss', default='', type=str, help='Losses for networks')
    parser.add_argument('--lw', default='', type=str, help='Weights for losses')
    
    return parser
    
def cfg_convert(config):
    if config['trset'] == 'SALOD':
        config['vals'] = ['SALOD']
    elif config['trset'] == 'simple':
        config['vals'] = ['tough', 'normal']
    elif config['vals'] == 'all':
        if config['trset'] == 'DUTS-TR':
            config['vals'] = ['PASCAL-S', 'ECSSD', 'HKU-IS', 'DUTS-TE', 'DUT-OMRON']
        if config['trset'] == 'COD-TR':
            config['vals'] = ['COD-TE', 'NC4K', 'CAMO-TE']
    else:
        config['vals'] = config['vals'].split(',')

    save_path = os.path.join(config['save_path'], config['model_name'], config['backbone'], config['sub'])
    check_path(save_path)
    config['save_path'] = save_path
    
    weight_path = os.path.join(config['weight_path'], config['model_name'], config['backbone'], config['sub'])
    check_path(weight_path)
    config['weight_path'] = weight_path
    
    train_split = config['train_split']
    if train_split == 10:
        config['batch'] = 1
        config['epoch'] = 500
        config['step_size'] = [400]
    elif train_split == 30:
        config['epoch'] = 900
        config['step_size'] = [600]
    elif train_split == 50:
        config['epoch'] = 600
        config['step_size'] = [400]
    elif train_split == 100:
        config['epoch'] = 450
        config['step_size'] = [300]
    elif train_split == 300:
        config['epoch'] = 300
        config['step_size'] = [200]
    elif train_split == 500:
        config['epoch'] = 150
        config['step_size'] = [100]
    elif train_split == 1000:
        config['epoch'] = 90
        config['step_size'] = [60]
    elif train_split == 3000:
        config['epoch'] = 60
        config['step_size'] = [40]
    elif train_split == 5000:
        config['epoch'] = 45
        config['step_size'] = [30]