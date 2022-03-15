import sys
import importlib
import torch
import os
from PIL import Image
from util import *
import numpy as np

from base.framework_factory import load_framework

mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])

def img_process(image, config):
    image = image.resize((config['size'], config['size']))
    image = np.array(image).astype(np.float32)
    image = ((image / 255.) - mean) / std
    image = image.transpose((2, 0, 1))
    im = torch.tensor(np.expand_dims(image, 0)).float()
    return im

def detect_img(img_path, model, config):
    image = Image.open(img_path).convert('RGB')
    image = img_process(image, config)
    
    Y = model(image.cuda())
    pred = Y['final'].sigmoid_().cpu().data.numpy()
    
    start = len(img_fold)
    im = Image.fromarray((pred[0, 0] * 255)).convert('L')
    #print(img_path[start:].split('.')[0])
    save_name = './photo/' + img_path[start:].split('.')[0] + '.png'
    fold_name = '/'.join(save_name.split('/')[:-1])
    if not os.path.exists(fold_name):
        os.makedirs(fold_name)
    im.save(save_name)
    return pred

# 迭代处理图片
def recursion(base_path, model, config):
    sub_list = os.listdir(base_path)
    for sub_name in sub_list:
        sub_path = os.path.join(base_path, sub_name)
        if os.path.isdir(sub_path):
            recursion(sub_path, model, config)
        elif sub_path.split('.')[-1] in ('png', 'jpg'):
            detect_img(sub_path, model, config)

#img_fold = '/media/data2/meiling/STMD/dataset/H36M-MultiView-test/'
#img_fold = '/media/data2/meiling/STMD/SYSU_Group/train/32/c1/addition/'
#img_fold = '/media/data2/meiling/STMD/SYSU_Group/train/32/'
#img_fold = '../dataset/ESOD/image/'
img_fold = './photo/'
def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    
    
    config, model, _, _, _, saver = load_framework(net_name)
    
    saved_model = torch.load(config['weight'], map_location='cpu')
    #new_name = {}
    #for k, v in saved_model.items():
    #    if k.startswith('model'):
    #        new_name[k[6:]] = v
    #    else:
    #        new_name[k] = v
    #model.load_state_dict(new_name)
    model.load_state_dict(saved_model)
    model.eval()
    model = model.cuda()
    
    recursion(img_fold, model, config)
    
if __name__ == "__main__":
    main()