import importlib
from torch import nn
import math

from torch.nn import functional as F
from torch.optim import *
import torch.optim.lr_scheduler as sche


# Base SGD
sgd_base_config = {
    'optim': 'SGD',
    'lr': 5e-3,
    'agg_batch': 32,
    'epoch': 40,
    }
def sgd_base(optimizer, current_iter, total_iter, config):
    if (current_iter / total_iter) < 0.5:
        factor = 1
    elif (current_iter / total_iter) < 0.75:
        factor = 0.1
    else:
        factor = 0.01
    #factor = 1 if (current_iter / total_iter) < 0.66 else 0.1
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']

# Base SGD poly
sgd_poly_config = {
    'optim': 'SGD',
    'lr': 1e-3,
    'agg_batch': 32,
    'epoch': 40,
    }
def sgd_poly(optimizer, current_iter, total_iter, config):
    factor = pow(1 - (current_iter / total_iter), 0.9)
        
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']

# Base Adam
adam_base_config = {
    'optim': 'Adam',
    'lr': 1e-4,
    'agg_batch': 32,
    'epoch': 40,
}
def adam_base(optimizer, current_iter, total_iter, config):
    if (current_iter / total_iter) < 0.5:
        factor = 1
    elif (current_iter / total_iter) < 0.75:
        factor = 0.1
    else:
        factor = 0.01
        
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']

# F3Net
sgd_f3net_config = {
    'optim': 'SGD',
    'lr': 5e-2,
    'agg_batch': 32,
    'epoch': 40,
    }
def sgd_f3net(optimizer, current_iter, total_iter, config):
    factor = (1 - abs(current_iter / total_iter * 2 - 1))
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']

# F3Net
sgd_pfsnet_config = {
    'optim': 'SGD',
    'lr': 5e-2,
    'agg_batch': 20,
    'epoch': 50,
    }
def sgd_pfsnet(optimizer, current_iter, total_iter, config):
    factor = (1 - abs(current_iter / total_iter * 2 - 1))
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']

# New test
sgd_scfnet_config = {
    'optim': 'SGD',
    'lr': 6.4e-2,
    'agg_batch': 128,
    'epoch': 69,
    'warmup': 5,
}
def sgd_scfnet(optimizer, current_iter, total_iter, config):
    min_lr = 6.4e-4
    max_lr = 6.4e-2
    mum_step = config['iter_per_epoch'] * config['warmup']
    # Warmup
    if config['cur_epoch'] < (config['warmup']+1):
        #factor = current_iter / (mum_step + 1e-8)
        lr = min_lr + abs(max_lr - min_lr) / (mum_step + 1e-8) * current_iter
    else:
        T_max = total_iter-mum_step
        cur_iter = current_iter-mum_step
        lr = (1 + math.cos(math.pi * cur_iter / T_max)) / (1 + math.cos(math.pi * (cur_iter - 1) / T_max)) * abs(optimizer.param_groups[1]['lr'] - min_lr) + min_lr
    
    optimizer.param_groups[0]['lr'] = lr * 0.1
    optimizer.param_groups[1]['lr'] = lr
    
# New test
sche_our_config = {
    'optim': 'SGD',
    'lr': 6.4e-2,
    'agg_batch': 128,
    'epoch': 69,
    'warmup': 0,
}
def sche_our(optimizer, current_iter, total_iter, config):
    min_lr = 6.4e-4
    max_lr = 6.4e-2
    mum_step = config['iter_per_epoch'] * config['warmup']
    # Warmup
    if config['cur_epoch'] < (config['warmup']+1):
        #factor = current_iter / (mum_step + 1e-8)
        lr = min_lr + abs(max_lr - min_lr) / (mum_step + 1e-8) * current_iter
    else:
        T_max = total_iter-mum_step
        cur_iter = current_iter-mum_step
        lr = (1 + math.cos(math.pi * cur_iter / T_max)) / (1 + math.cos(math.pi * (cur_iter - 1) / T_max)) * abs(optimizer.param_groups[1]['lr'] - min_lr) + min_lr
    
    optimizer.param_groups[0]['lr'] = lr * 0.1
    optimizer.param_groups[1]['lr'] = lr
    


# New test
sche_new_config = {
    'optim': 'AdamW',
    'lr': 2e-4,
    'agg_batch': 16,
    'epoch': 30,
}
def sche_new(optimizer, current_iter, total_iter, config):
    factor = pow((1 - 1.0 * current_iter / total_iter), 0.9)
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']


sgd_menet_config = {
    'optim': 'SGD',
    'lr': 2.5e-2,
    'agg_batch': 24,
    'epoch': 40,
}
def sgd_menet(optimizer, current_iter, total_iter, config):
    factor = (1-abs((current_iter+1)/(total_iter+1)*2-1))
    optimizer.param_groups[0]['lr'] = factor * config['lr'] * 0.1
    optimizer.param_groups[1]['lr'] = factor * config['lr']



def Strategy(model, config):
    strategy = config['strategy']
    stra_config = eval(strategy + '_config')
    config.update(stra_config)
    
    if 'params' in config.keys():
        module_lr = [{'params' : getattr(model, p[0]).parameters(), 'lr' : p[1]} for p in config['params']]
    else:
        encoder = []
        others = []
        for param in model.named_parameters():
            if 'encoder.' in param[0]:
                encoder.append(param[1])
            else:
                others.append(param[1])
        if len(encoder) == 0:
            print("Warning: parameters in encoder not found!")
        module_lr = [{'params' : encoder, 'lr' : config['lr']*0.1}, {'params' : others, 'lr' : config['lr']}]
        
    optim = config['optim']
    if optim == 'SGD':
        optimizer = SGD(params=module_lr, lr=config['lr'], momentum=0.9, weight_decay=0.0005)
    elif optim == 'Adam':
        optimizer = Adam(params=module_lr, lr=config['lr'], weight_decay=0.0005)
    elif optim == 'AdamW':
        optimizer = AdamW(params=module_lr, lr = config['lr'], weight_decay=0.05)
        
    schedule = eval(strategy)
    return optimizer, schedule
