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

def get_image_list(name, config):
    image_root = os.path.join(config['data_path'], name, 'images')
    gt_root = os.path.join(config['data_path'], name, 'segmentations')
    
    images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg')])
    gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')])
    
    return images, gts

def pad_image(image):
    h, w = image.size
    new_h = max((h + 31) // 32 * 32, 128) 
    new_w = max((w + 31) // 32 * 32, 128)
    pad_h = new_h - h
    pad_w = new_w - w
    
    img = np.array(image)
    pads = [(0, pad_w), (0, pad_h)]
    if len(img.shape) == 3:
        pads.append((0, 0))
    img = np.pad(img, pads, 'constant', constant_values=0)
    #image = Image.fromarray(img)
    
    return Image.fromarray(img)

#def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
def get_loader(name, config=None):
    dataset = Train_Dataset(config['trset'], config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config['batch'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
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

class Train_Dataset(data.Dataset):
    def __init__(self, name, config):
        self.config = config
        self.images, self.gts = get_image_list(name, config)
        self.size = len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        
        if self.config['data_aug']:
            image, gt = rotate(image, gt)
        
        if not self.config['orig_size']:
            img_size = self.config['size']
            image = image.resize((img_size, img_size))
            gt = gt.resize((img_size, img_size))
        
        image = np.array(image).astype(np.float32)
        gt = np.array(gt)
        
        #print(image.shape, gt.shape)
        if random.random() > 0.5:
            image = image[:, ::-1]
            gt = gt[:, ::-1]
        
        
        if self.config['data_aug']:
            #print('random light')
            image = random_light(image)
        
        image = ((image / 255.) - mean) / std
        #image = image - mean
        image = image.transpose((2, 0, 1))
        gt = np.expand_dims((gt > 128).astype(np.float32), axis=0)
        return image, gt

    def __len__(self):
        return self.size


class Val_Dataset:
    def __init__(self, name, config=None):
        st = time.time()
        
        self.config = config
        self.image_paths, self.gt_paths = get_image_list(name, config)
        self.size = len(self.image_paths)
        
        bar = Bar('Dataset {:10}:'.format(name), max=self.size)
        
        self.images = []
        self.gts = []
        self.names = []
        
        img_size = config['size']
        
        for i, image_name, gt_name in zip(range(self.size), self.image_paths, self.gt_paths):
            image = np.array(Image.open(self.image_paths[i]).convert('RGB').resize((img_size, img_size))).astype(np.float32)
            gt = np.array(Image.open(self.gt_paths[i]).convert('L').resize((img_size, img_size)))
            img_name = self.image_paths[i].split('/')[-1].split('.')[0]
            
            if self.config['orig_size']:
                pass
                #image = pad_image(image)
            
            image = ((image / 255.) - mean) / std
            #image = image - mean
            image = image.transpose((2, 0, 1))
            gt = (gt > 128).astype(np.float32)
            self.images.append(image)
            self.gts.append(gt)
            self.names.append(img_name)
            
            Bar.suffix = '{}/{}, using time: {}s.'.format(i, self.size, round(time.time() - st, 1))
            bar.next()

        bar.finish()
        #print('Loading dataset {:10} using {}s.'.format(name, round(time.time() - st, 1)))
        
        self.images = np.array(self.images)
        self.gts = np.array(self.gts)

    def load_all_data(self):
        return self.images, self.gts, self.names

        
class Test_Dataset:
    def __init__(self, name, config=None):
        self.config = config
        self.images, self.gts = get_image_list(name, config)
        self.size = len(self.images)

    def load_data(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        if not self.config['orig_size']:
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
    config = {'orig_size': True, 'size': 288, 'data_path': '../dataset/'}
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