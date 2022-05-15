import sys
import argparse  
import os
from util import *

def base_config(strategy):
    '''
    if cfg_dict == None:
        cfg_dict = {
            'optim': 'SGD',
            'schedule': 'Pyramid',
            'lr': 0.05,
            'batch': 32,
            'ave_batch': 1,
            'epoch': 40,
            'step_size': '20',
            'gamma': 0.1,
        }
    '''

    parser = argparse.ArgumentParser()

    # Training schedule
    parser.add_argument('model_name', default='', help='Training model')
    parser.add_argument('--strategy', default=strategy, help='Training strategy')
    #parser.add_argument('--optim', default=cfg_dict['optim'], help='set the optimizer of model [Adam or SGD]')
    #parser.add_argument('--schedule', default=cfg_dict['schedule'], help='set the scheduler')
    #parser.add_argument('--lr', default=cfg_dict['lr'], type=float, help='set base learning rate')
    #parser.add_argument('--batch', default=cfg_dict['batch'], type=int, help='Batch Size for dataloader')
    #parser.add_argument('--epoch', default=cfg_dict['epoch'], type=int, help='Training epoch')
    #parser.add_argument('--step_size', default=cfg_dict['step_size'], type=str, help='Lr decrease steps')
    #parser.add_argument('--gamma', default=cfg_dict['gamma'], type=float)
    parser.add_argument('--ave_batch', default=1, type=int, help='Number of batches for Backpropagation')
    
    parser.add_argument('--backbone', default='resnet', help='Set the backbone of the model')
    parser.add_argument('--clip_gradient', default=0, type=float, help='Max gradient')
    parser.add_argument('--test_batch', default=1, type=int, help='Batch Size for Testing')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--consist', action='store_true')
    parser.add_argument('--sub', default='base', help='Job name')
    
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpus', default='0', type=str, help='Set the cuda devices')
    
    
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--multi', action='store_true')
    
    # Data setting
    parser.add_argument('--size', default=320, type=int, help='Input size')
    parser.add_argument('--trset', default='SALOD', help='Set the traing set') # SALOD, DUTS-TE, simple
    parser.add_argument('--vals', default='all', help='Set the testing sets')
    parser.add_argument('--data_path', default='../dataset', help='The name of network')
    parser.add_argument('--save_path', default='./result/', help='Save path of network')
    parser.add_argument('--weight_path', default='./weight/', help='Weight path of network')
    
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--show_param', action='store_true')

    # testing
    parser.add_argument('--weight', default='', help='Trained weight path')
    parser.add_argument('--save', action='store_true')
    
    # Use for SALOD dataset
    parser.add_argument('--train_split', default=10000, type=int, help='Use for ESOD dataset')
    
    
    # Construct loss by loss_factory. More details in methods\base\loss.py.
    # loss: 'b': BCE, 's': SSIM, 'i': IOU, 'd': DICE, 'e': Edge, 'c': CTLoss
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
    
    #if config['step_size'] != '':
    #    config['step_size'] = [int(ss) for ss in config['step_size'].split(',')]

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