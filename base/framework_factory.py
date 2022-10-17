import os, sys
import torch
import numpy as np
import importlib
import timm

def build_network(net_name, config):
    all_names = timm.list_models(pretrained=True)
    if config['backbone'] not in all_names:
        print(f"Backbone \"{config['backbone']}\" is not available now. Please use one of backbone in {all_names}.")
        sys.exit()
    else:
        encoder = timm.create_model(config['backbone'], features_only=True, pretrained=True)
        fl = encoder.feature_info.channels()
        model = importlib.import_module('methods.{}.model'.format(net_name)).Network(config, encoder, fl)
        return model

def load_framework(net_name):
    # Load Configure
    config = importlib.import_module('methods.{}.config'.format(net_name)).get_config()
    #print(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus']
    
    # Constructing network
    model = build_network(net_name, config) #importlib.import_module('base.model').Network(net_name, config)
    
    if config['show_param']:
        from thop import profile
        input = torch.randn(1, 3, config['size'], config['size'])
        macs, params = profile(model, inputs=(input, ))
        params = np.sum([p.numel() for p in model.parameters()]).item()
        print('MACs: {:.2f} G, Params: {:.2f} M.'.format(macs / 1e9, params / 1e6))
        
    loss = importlib.import_module('base.loss').Loss_factory(config['loss'])
    
    # Loading Saver if it exists
    if config['save'] and os.path.exists('methods/{}/saver.py'.format(net_name)):
        saver = importlib.import_module('methods.{}.saver'.format(net_name)).Saver
    else:
        saver = None
    
    config['num_gpu'] = len(config['gpus'].split(','))
    
    if config['num_gpu'] > 1:
        gpus = range(config['num_gpu'])
        model = torch.nn.DataParallel(model, device_ids=gpus)
    model = model.cuda()
    
    # Set optimizer and schedule
    optimizer, schedule = importlib.import_module('base.strategy').Strategy(model, config)
    
    return config, model, optimizer, schedule, loss, saver
