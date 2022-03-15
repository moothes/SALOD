import sys
import importlib
from data import Test_Dataset
from thop import profile
import torch
from torch.optim import SGD, Adam
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image
from util import *
import numpy as np

import argparse

from base.framework_factory import load_framework
from metric import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


test_imgs = []
with open('../dataset/ESOD/test.txt', 'r') as f:
    for line in f.readlines():
        test_imgs.append(line.strip())
        
#test_imgs = test_imgs[:100]
        
        
hard = {}
normal = {}
easy = {}
with open('../dataset/ESOD/new_list.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        name, score = line.strip().split(' ')
        if name in test_imgs:
            if i < 10000:
                hard[name] = [score, ]
            elif i < 27930:
                normal[name] = [score, ]
            else:
                easy[name] = [score, ]
                
print(len(hard.keys()), len(normal.keys()))
        
gt_path = '../dataset/ESOD/mask/'
pred_path = './temp1/'

MR = MetricRecorder(len(test_imgs))
scores = []
hidxs = []
nidxs = []
eidxs = []
for i, pred_tag in enumerate(test_imgs):
    pred_name = pred_tag + '.png'
    # = pred_name.split('.')[0]
    pred = Image.open(pred_path + pred_name).convert('L')
    
    gt = np.array(Image.open(gt_path + pred_tag + '.png').convert('L'))
    out_shape = gt.shape
    pred = np.array(pred.resize((out_shape[::-1])))
    
    if i % 1000 == 0:
        print(i)
    
    #print(pred.shape, gt.shape)
    pred, gt = normalize_pil(pred, gt)
    MR.update(pre=pred, gt=gt)
    
    if pred_tag in hard.keys():
        hidxs.append(i)
        scores.append(hard[pred_tag])
    elif pred_tag in normal.keys():
        nidxs.append(i)
        scores.append(normal[pred_tag])
    elif pred_tag in easy.keys():
        eidxs.append(i)
        scores.append(easy[pred_tag])

pre = MR.fm.precision
rec = MR.fm.recall
fscore = pre * rec * 1.3 / (0.3 * pre + rec + 1e-8)
fmax = np.max(fscore, axis=1, keepdims=True)

print(len(hidxs), len(nidxs), len(eidxs))

hs = fmax[hidxs]
ns = fmax[nidxs]
es = fmax[eidxs]

print(np.mean(hs), np.mean(ns), np.mean(es))
# 7292 13226 7412
#print(hs.shape, ns.shape, es.shape)
#print(np.array(scores).shape, fmax.shape)

sort_fmax = np.sort(fmax, axis=0)
#print(sort_fmax)
print(np.mean(sort_fmax[:7292]), np.mean(sort_fmax[7292:-7412]), np.mean(sort_fmax[-7412:]))
print(np.mean(sort_fmax))
'''
xy = np.zeros((len(scores), 2))

for i, fm, score in zip(range(len(scores)), fmax, scores):
    if i % 1000 == 0:
        print(i)
    xy[i, 0] = score[0]
    xy[i, 1] = fm[0]
    #print(area)
plt.scatter(xy[:, 0], xy[:, 1], s=1, marker='o')
#plt.scatter(xy[:, 0], xy[:, 1], s=3, 'blue', 'o')
plt.savefig('./aaa.png')
    
#print(MR.fm.precision.shape, MR.fm.recall.shape)
mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)

print('  MAE: {}, Max-F: {}, Maen-F: {}, SM: {}, EM: {}, Fbw: {}.'.format(mae, maxf, meanf, sm, em, wfm))
'''