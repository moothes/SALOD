import os, glob, random
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import random
import time
from progress.bar import Bar
import cv2

#mean = np.array((104.00699, 116.66877, 122.67892)).reshape((1, 1, 3))
mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])



def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def get_image_list(name, config, phase):
    images = []
    gts = []
    #print(name)
    if name in ('simple', 'tough', 'normal'):
        train_split = 10000
        
        print('Objectness shifting experiment.')
        # Objectness
        list_file = 'clean_list.txt'
        #print(list_file)
        #f = open(os.path.join(config['data_path'], 'SALOD/gaps.txt'), 'r')
        #f = open(os.path.join(config['data_path'], 'SALOD/new_list.txt'), 'r')
        #f = open(os.path.join(config['data_path'], 'SALOD/place_list.txt'), 'r')
        f = open(os.path.join(config['data_path'], 'SALOD/{}'.format(list_file)), 'r')
        if name == 'simple':
            img_list = f.readlines()[-train_split:]
        elif name == 'normal':
            img_list = f.readlines()[train_split:-train_split]
        else:
            img_list = f.readlines()[:train_split]
                
        for i in range(len(img_list)):
            img_list[i] = img_list[i].split(' ')[0]
            #print(img_list[i])
                
        images = [os.path.join(config['data_path'], 'SALOD/images', line.strip() + '.jpg') for line in img_list]
        gts = [os.path.join(config['data_path'], 'SALOD/mask', line.strip() + '.png') for line in img_list]
        
        # Benchmark + few_shot
    elif name == 'SALOD':
        '''
        if phase == 'train':
            f = open(os.path.join(config['data_path'], 'SALOD/train.txt'), 'r')
            img_list = f.readlines()[:config['train_split']]
        else:
            f = open(os.path.join(config['data_path'], 'SALOD/test.txt'), 'r')
            img_list = f.readlines()
        '''
        
        f = open(os.path.join(config['data_path'], 'SALOD/{}.txt'.format(phase)), 'r')
        img_list = f.readlines()
        
        
        images = [os.path.join(config['data_path'], name, 'images', line.strip() + '.jpg') for line in img_list]
        gts = [os.path.join(config['data_path'], name, 'mask', line.strip() + '.png') for line in img_list]
    else:
        image_root = os.path.join(config['data_path'], name, 'images')
        '''
        if name == 'DUTS-TR' and phase == 'train':
            tag = 'segmentations' #'crf2' # 'pseudo'
        elif name == 'MSB-TR' and phase == 'train':
            tag = 'crf3'
            #tag = 'iter1'
        else:
            tag = 'segmentations'
        print(tag)
        '''
        
        gt_root = os.path.join(config['data_path'], name, 'segmentations')
        #print(gt_root)
        
        images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')])
        gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
        #print(len(images), len(gts))
    
    return images, gts

def get_loader(config):
    dataset = Train_Dataset(config['trset'], config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader

def random_light(x):
    contrast = np.random.rand(1)+0.5
    light = np.random.randint(-20,20)
    x = contrast*x + light
    return np.clip(x,0,255)

def rotate(img, gt):
    angle = np.random.randint(-25,25)
    img = img.rotate(angle)
    gt = gt.rotate(angle)
    return img, gt


def RandomCrop(image, mask):
    H, W = image.size
    randw = np.random.randint(W/8)
    randh = np.random.randint(H/8)
    offseth = 0 if randh == 0 else np.random.randint(randh)
    offsetw = 0 if randw == 0 else np.random.randint(randw)
    p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
    #return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]
    return image.crop((p0, p2, p1, p3)), mask.crop((p0, p2, p1, p3))


class Train_Dataset(data.Dataset):
    def __init__(self, name, config):
        self.config = config
        self.images, self.gts = get_image_list(name, config, 'train')
        self.size = len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        
        #print('orig: ', image.size, gt.size)
        if self.config['data_aug']:
            #image, gt = rotate(image, gt)
            #image = random_light(image)
            image, gt = RandomCrop(image, gt)
        #print('croped: ', image.size, gt.size)
        
        img_size = self.config['size']
        image = image.resize((img_size, img_size))
        gt = gt.resize((img_size, img_size))
    
        image = np.array(image).astype(np.float32)
        gt = np.array(gt)
        
        #print(image.shape, gt.shape)
        if random.random() > 0.5:
            image = image[:, ::-1]
            gt = gt[:, ::-1]
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        gt = np.expand_dims((gt > 128).astype(np.float32), axis=0)
        return image, gt

    def __len__(self):
        return self.size

class Test_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        self.images, self.gts = get_image_list(name, config, 'test')
        self.size = len(self.images)

    def load_data(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        #if not self.config['orig_size']:
        image = image.resize((self.config['size'], self.config['size']))
        image = np.array(image).astype(np.float32)
        gt = np.array(Image.open(self.gts[index]).convert('L'))
        name = self.images[index].split('/')[-1].split('.')[0]
        
        
        image = ((image / 255.) - mean) / std
        image = image.transpose((2, 0, 1))
        image = torch.tensor(np.expand_dims(image, 0)).float()
        gt = (gt > 128).astype(np.float32)
        return image, gt, name

def test_data():
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset'}
    dataset = 'SOD'
    
    '''
    data_loader = Test_Dataset(dataset, config)
    #data_loader = Train_Dataset(dataset, config)
    data_size = data_loader.size
    
    for i in range(data_size):
        img, gt, name = data_loader.load_data(i)
        #img, gt = data_loader.__getitem__(i)
        new_img = (img * std + mean) * 255.
        #new_img = gt * 255
        print(np.min(new_img), np.max(new_img))
        new_img = (new_img).astype(np.uint8)
        #print(new_img.shape).astype(np.)
        im = Image.fromarray(new_img)
        #im.save('temp/' + name + '.jpg')
        im.save('temp/' + str(i) + '.jpg')
    
    '''
    
    data_loader = Val_Dataset(dataset, config)
    imgs, gts, names = data_loader.load_all_data()
    print(imgs.shape, gts.shape, len(names))
    

if __name__ == "__main__":
    test_data()