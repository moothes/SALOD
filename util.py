import os
import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import importlib
import scipy
import scipy.ndimage
from torch.optim import SGD, Adam
import torch


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def freeze_bn(model):
    for m in model.base.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def label_edge_prediction(label):
    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
    fx = np.reshape(fx, (1, 1, 3, 3))
    fy = np.reshape(fy, (1, 1, 3, 3))
    fx = Variable(torch.from_numpy(fx)).cuda()
    fy = Variable(torch.from_numpy(fy)).cuda()
    contour_th = 1.5
    
    # convert label to edge
    label = label.float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad

def label_edge_prediction_new(label):
    ero = 1 - F.max_pool2d(1 - label, kernel_size=5, stride=1, padding=2)  # erosion
    dil = F.max_pool2d(label, kernel_size=5, stride=1, padding=2)            # dilation
    
    edge = dil - ero
    return edge

def mask_normalize(mask):
    return mask/(np.amax(mask)+1e-8)

def compute_mae(mask1,mask2):
# input 'mask1': HxW or HxWxn (asumme that all the n channels are the same and only the first channel will be used)
#       'mask2': HxW or HxWxn
# output: a value MAE, Mean Absolute Error
    if(len(mask1.shape)<2 or len(mask2.shape)<2):
        print("ERROR: Mask1 or mask2 is not matrix!")
        exit()
    if(len(mask1.shape)>2):
        mask1 = mask1[:,:,0]
    if(len(mask2.shape)>2):
        mask2 = mask2[:,:,0]
    if(mask1.shape!=mask2.shape):
        print("ERROR: The shapes of mask1 and mask2 are different!")
        exit()

    h,w = mask1.shape[0],mask1.shape[1]
    mask1 = mask_normalize(mask1)
    mask2 = mask_normalize(mask2)
    sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    maeError = sumError/(float(h)*float(w)+1e-8)

    return maeError

def compute_pre_rec(gt, mask, mybins=np.arange(0,256)):
    if(len(gt.shape)<2 or len(mask.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(mask.shape)>2): # convert to one channel
        mask = mask[:,:,0]
    if(gt.shape!=mask.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
    pp = mask[gt>128] # mask predicted pixel values in the ground truth foreground region
    nn = mask[gt<=128] # mask predicted pixel values in the ground truth bacground region

    #print(np.max(mask), np.min(mask))
    pp_hist,pp_edges = np.histogram(pp, bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist,nn_edges = np.histogram(nn, bins=mybins)

    pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)
    #print(pp_hist_flip_cum, nn_hist_flip_cum + pp_hist_flip_cum)
    #print(pp_hist_flip_cum)
    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)
    #print(pp_hist_flip_cum, gtNum)
    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision,(len(precision))),np.reshape(recall,(len(recall)))
    
def eval_mae(mask1, mask2):
# input 'mask1': HxW or HxWxn (asumme that all the n channels are the same and only the first channel will be used)
#       'mask2': HxW or HxWxn
# output: a value MAE, Mean Absolute Error
    if(len(mask1.shape)<2 or len(mask2.shape)<2):
        print("ERROR: Mask1 or mask2 is not matrix!")
        exit()
    if(len(mask1.shape)>2):
        mask1 = mask1[:,:,0]
    if(len(mask2.shape)>2):
        mask2 = mask2[:,:,0]
    if(mask1.shape!=mask2.shape):
        print("ERROR: The shapes of mask1 and mask2 are different!")
        exit()

    h,w = mask1.shape[0],mask1.shape[1]
    mask1 = mask_normalize(mask1)
    mask2 = mask_normalize(mask2)
    sumError = np.sum(np.absolute((mask1.astype(float) - mask2.astype(float))))
    maeError = sumError/(float(h)*float(w)+1e-8)

    return maeError

def eval_f(pred, gt, mybins=np.arange(0,256)):
    if(len(gt.shape)<2 or len(pred.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(pred.shape)>2): # convert to one channel
        pred = pred[:,:,0]
    if(gt.shape!=pred.shape):
        print("ERROR: The shapes of gt and mask are different!")
        exit()

    gtNum = gt[gt>128].size # pixel number of ground truth foreground regions
    pp = pred[gt>128] # mask predicted pixel values in the ground truth foreground region
    nn = pred[gt<=128] # mask predicted pixel values in the ground truth bacground region

    #print(np.max(mask), np.min(mask))
    pp_hist,pp_edges = np.histogram(pp, bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist,nn_edges = np.histogram(nn, bins=mybins)

    pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)
    #print(pp_hist_flip_cum, nn_hist_flip_cum + pp_hist_flip_cum)
    #print(pp_hist_flip_cum)
    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)
    #print(pp_hist_flip_cum, gtNum)
    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision,(len(precision))), np.reshape(recall,(len(recall)))
        
def eval_f_new(pred, gt):
    if torch.mean(gt) == 0.0:
        return None

    prec, recall = self._eval_pr(pred, gt, 255)
    f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
    f_score[f_score != f_score] = 0 # for Nan
    avg_f += f_score

    return avg_f
    

def _eval_pr(pred, gt, num):
    if self.cuda:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall
        
def eval_fbw(pred, gt):
    beta2 = 0.3
    if np.mean(gt) == 0: # the ground truth is totally black
        return 1 - np.mean(pred)
    else:
        if not np.all(np.isclose(gt, 0) | np.isclose(gt, 1)):
            raise ValueError("'gt' must be a 0/1 or boolean array")
        gt_mask = np.isclose(gt, 1)
        not_gt_mask = np.logical_not(gt_mask)

        E = np.abs(pred - gt)
        dist, idx = scipy.ndimage.morphology.distance_transform_edt(not_gt_mask, return_indices=True)

        # Pixel dependency
        Et = np.array(E)
        # To deal correctly with the edges of the foreground region:
        Et[not_gt_mask] = E[idx[0, not_gt_mask], idx[1, not_gt_mask]]
        sigma = 5.0
        EA = scipy.ndimage.gaussian_filter(Et, sigma=sigma, truncate=3 / sigma, mode='constant', cval=0.0)
        min_E_EA = np.minimum(E, EA, where=gt_mask, out=np.array(E))

        # Pixel importance
        B = np.ones(gt.shape)
        B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])
        Ew = min_E_EA * B

        # Final metric computation
        eps = np.spacing(1)
        TPw = np.sum(gt) - np.sum(Ew[gt_mask])
        FPw = np.sum(Ew[not_gt_mask])
        R = 1 - np.mean(Ew[gt_mask])  # Weighed Recall
        P = TPw / (eps + TPw + FPw)  # Weighted Precision

        # Q = 2 * (R * P) / (eps + R + P)  # Beta=1
        return (1 + beta2) * (R * P) / (eps + R + (beta2 * P))

def eval_e(pred, gt, num=255):
    score = np.zeros(num)
    thlist = np.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (pred >= thlist[i]).astype(np.float32)
        if np.mean(gt) == 0.0: # the ground-truth is totally black
            y_pred_th = - y_pred_th
            enhanced = y_pred_th + 1
        elif np.mean(gt) == 1.0: # the ground-truth is totally white
            enhanced = y_pred_th
        else: # normal cases
            fm = y_pred_th - np.mean(y_pred_th)
            gt = gt - np.mean(gt)
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4

        #print(gt.shape)
        score[i] = np.sum(enhanced) / (gt.size - 1 + 1e-20)

    return score
    
def eval_s(pred, gt):
    gt = gt > 0.5

    y = np.mean(gt)
    #print(y)
    if y == 0:
        x = np.mean(pred)
        Q = 1 - x
    elif y == 1:
        x = np.mean(pred)
        Q = x
    else:
        alpha = 0.5
        so = S_object(pred, gt)
        sr = S_region(pred, gt)
        #print(so, sr)
        Q = alpha * so + (1 - alpha) * sr
    Q = 0 if Q < 0 else Q

    return Q
    

eps = 2.2204e-16

def ssim(pred, gt):
    size = pred.size
    
    x = np.mean(pred)
    y = np.mean(gt)
    
    sig_x = np.sum((pred - x) ** 2) / (size - 1 + eps)
    sig_y = np.sum((gt - y) ** 2) / (size - 1 + eps)
    sig_xy = np.sum((pred - x) * (gt - y)) / (size - 1 + eps)
    
    alpha = 4 * x * y * sig_xy
    beta = (x**2 + y**2) * (sig_x + sig_y)
    
    #print(x, y, sig_x, sig_y, sig_xy, alpha, beta)
    
    if alpha != 0:
        Q = alpha / (beta + eps)
    elif alpha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q
    

def divide(img, x, y):
    h, w = img.shape
    size = w * h
    
    lt = img[:y, :x]
    rt = img[:y, x:]
    lb = img[y:, :x]
    rb = img[y:, x:]
    
    w1 = x * y / size
    w2 = (w - x) * y / size
    w3 = (x * (h - y)) / size
    w4 = 1 - w1 - w2 - w3
    
    return lt, rt, lb, rb, w1, w2, w3, w4
    

def centroid(img):
    h, w = img.shape
    
    total = np.sum(img)
    if total == 0:
        X = w // 2
        Y = h // 2
    else:
        i = range(1, w + 1)
        j = range(1, h + 1)
        X = int(np.sum(np.sum(img, axis=0) * i) / total + 0.5)
        Y = int(np.sum(np.sum(img, axis=1) * j) / total + 0.5)
    
    return X, Y

def obj(pred, gt):
    x = np.mean(pred[gt == 1])
    sigma_x = np.std(pred[gt == 1])
    #print(np.max(gt))

    score = 2.0 * x / (x**2 + 1.0 + sigma_x + eps)
    return score

def S_object(pred, gt):
    p_fg = pred * gt
    O_FG = obj(p_fg, gt)
    
    p_bg = (1 - pred) * (1 - gt)
    O_BG = obj(p_bg, 1 - gt)
    
    u = np.mean(gt)
    Q = u * O_FG + (1 - u) * O_BG
    #print(O_FG, O_BG)
    return Q

def S_region(pred, gt):
    X, Y = centroid(gt)
    
    ltg, rtg, lbg, rbg, w1, w2, w3, w4 = divide(gt, X, Y)
    ltp, rtp, lbp, rbp, _, _, _, _ = divide(pred, X, Y)
    
    Q1 = ssim(ltp, ltg)
    Q2 = ssim(rtp, rtg)
    Q3 = ssim(lbp, lbg)
    Q4 = ssim(rbp, rbg)
    
    #print(X, Y)
    
    #print(w1, w2, w3, w4, Q1, Q2, Q3, Q4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

