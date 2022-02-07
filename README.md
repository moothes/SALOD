# SALOD

A benchmark for SALient Object Detection (SALOD) task. 

We re-implement 14 networks using same settings, including input size, data loader and evaluation metrics (thanks to [Metrics](https://github.com/lartpang/Py-SOD-VOS-EvalToolkit)). Hyperparameters for optimizer are different because of various network structures and objective functions. We try our best to tune the optimizer for these models to achieve the best performance one-by-one. Some other networks are debugging now, it is welcome for you tuning these networks to obtain better performance and contact us.

## Properties
1. **A unify interface for new models.** To develop a new network, you only need to 1) set configs; 2) define network; 3) define loss function.
2. We use the **Extend Salient Object Detection (ESOD) dataset**, which is a collection of several prevalent datasets, to evaluate SOD networks. 
3. Easy to adopt different backbones **(Available backbones: ResNet-50, VGG-16, MobileNet-v2, EfficientNet-B0, GhostNet, Res2Net)**
4. **Testing all networks on your own device.** By input the name of network, you can test old networks in this benchmark. Comparisons includes FPS, GFLOPs, model size and so on.
5. We implement a **loss factory** that you can change the loss functions for any metwork by passing different parameters.

## Methods:

 Methods | Publish. | Input | Weight | Optim. | LR    | Epoch | Paper | Src Code
 ----    | -----    | ----- | ------ | ------ | ----- | ----- | ----- | ------
 DHSNet  | CVPR2016 | 320^2 | 95M    | Adam   | 2e-5  | 30    | [openaccess](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf) | [Pytorch](https://github.com/xsxszab/DHSNet-Pytorch)  
 NLDF    | CVPR2017 | 320^2 | 161M   | Adam   | 1e-5  | 30    | [openaccess](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Non-Local_Deep_Features_CVPR_2017_paper.pdf) | [Pytorch](https://github.com/AceCoooool/NLDF-pytorch)/[TF](https://github.com/zhimingluo/NLDF) 
 Amulet  | ICCV2017 | 320^2 | 312M   | Adam   | 1e-5  | 30    | [openaccess](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Amulet_Aggregating_Multi-Level_ICCV_2017_paper.pdf) | [Pytorch](https://github.com/xsxszab/Amulet-Pytorch)  
 SRM     | ICCV2017 | 320^2 | 240M   | Adam   | 5e-5  | 30    | [openaccess](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_A_Stagewise_Refinement_ICCV_2017_paper.pdf) | [Pytorch](https://github.com/xsxszab/SRM-Pytorch) 
 PicaNet | CVPR2018 | 320^2 | 464M   | SGD    | 1e-2  | 30    | [openaccess](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_PiCANet_Learning_Pixel-Wise_CVPR_2018_paper.pdf) | [Pytorch](https://github.com/Ugness/PiCANet-Implementation)  
 DSS     | TPAMI2019| 320^2 | 525M   | Adam   | 2e-5  | 30    | [IEEE](https://ieeexplore.ieee.org/document/8315520/)/[ArXiv](https://arxiv.org/abs/1611.04849) | [Pytorch](https://github.com/AceCoooool/DSS-pytorch)  
 BASNet  | CVPR2019 | 320^2 | 374M   | Adam   | 1e-5  | 25    | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/NathanUA/BASNet)  
 CPD     | CVPR2019 | 320^2 | 188M   | Adam   | 1e-5  | 30    | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Cascaded_Partial_Decoder_for_Fast_and_Accurate_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/wuzhe71/CPD)  
 PoolNet | CVPR2019 | 320^2 | 267M   | Adam   | 5e-5  | 30    | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_A_Simple_Pooling-Based_Design_for_Real-Time_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/backseason/PoolNet)  
 EGNet   | ICCV2019 | 320^2 | 437M   | Adam   | 5e-5  | 30    | [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf) | [Pytorch](https://github.com/JXingZhao/EGNet)  
 SCRN    | ICCV2019 | 320^2 | 100M   | SGD    | 1e-2  | 30    | [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf) | [Pytorch](https://github.com/wuzhe71/SCRN)  
 GCPA    | AAAI2020 | 320^2 | 263M   | SGD    | 1e-2  | 30    | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6633) | [Pytorch](https://github.com/JosephChenHub/GCPANet)  
 ITSD    | CVPR2020 | 320^2 | 101M   | SGD    | 5e-3  | 30    | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/moothes/ITSD-pytorch)  
 MINet   | CVPR2020 | 320^2 | 635M   | SGD    | 1e-3  | 30    | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/lartpang/MINet)  
 `Tuning`  | -----    | ----- | ------ | ------ | ----- | ----- | ----- | -----
 *PAGE    | CVPR2019 | 320^2 | ------ | ------ | ----- | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Salient_Object_Detection_With_Pyramid_Attention_and_Salient_Edges_CVPR_2019_paper.pdf) | [TF](https://github.com/wenguanwang/PAGE-Net)  
 *PFA     | CVPR2019 | 320^2 | ------ | ------ | ----- | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/dizaiyoufang/pytorch_PFAN)  
 *F3Net   | AAAI2020 | 320^2 | ------ | ------ | ----- | ----- | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6916) | [Pytorch](https://github.com/weijun88/F3Net)  
 *PFPN   | AAAI2020 | 320^2 | ------ | ------ | ----- | ----- |  [aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/6892) | [Pytorch](https://github.com/chenquan-cq/PFPN)
 *LDF    | CVPR2020 | 320^2 | ------ | ------ | ----- | ----- |  [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Label_Decoupling_Framework_for_Salient_Object_Detection_CVPR_2020_paper.html)|  [Pytorch](https://github.com/weijun88/LDF)
 
 ## Usage
 
 ```
 # model_name: lower-cased method name. E.g. poolnet, egnet, gcpa, dhsnet or minet.
 python3 train.py model_name --gpus=0
 
 python3 test.py model_name --gpus=0 --weight=path_to_weight 
 
 python3 test_fps.py model_name --gpus=0
 
 # To evaluate generated maps:
 python3 eval.py --pre_path=path_to_maps
 ```
 
 
 
 ## Results
 
 We report benchmark results here. More results please refer to [Reproduction](https://github.com/moothes/SALOD/blob/master/readme/Reproduction.md), [Few-shot](https://github.com/moothes/SALOD/blob/master/readme/Few-shot.md) or [Generalization](https://github.com/moothes/SALOD/blob/master/readme/Generazation.md).
 
 **Notice: please contact us if you get better results.**
 
### VGG16-based:
Methods | #Param. | GFLOPs | Tr. Time | FPS  | max-F | ave-F | Fbw  | MAE  | SM   | EM    | Weight
 ----   | ---     | -----  | -----    | ---- | ----- | ----- | ---- | ---- | ---- | ----- | ----     
DHSNet  | 15.4    | 52.5   | 7.5      | 69.8 | .884  | .815  | .812 | .049 | .880 | .893  |      
Amulet  | 33.2    | 1362   | 12.5     | 35.1 | .855  | .790  | .772 | .061 | .854 | .876  |    
NLDF    | 24.6    | 136    | 9.7      | 46.3 | .886  | .824  | .828 | .045 | .881 | .898  |   
SRM     | 37.9    | 73.1   | 7.9      | 63.1 | .857  | .779  | .769 | .060 | .859 | .874  |   
PicaNet | 26.3    | 74.2   | 40.5*    | 8.8  | .889  | .819  | .823 | .046 | .884 | .899  |  
DSS     | 62.2    | 99.4   | 11.3     | 30.3 | .891  | .827  | .826 | .046 | .888 | .899  |    
BASNet  | 80.5    | 114.3  | 16.9     | 32.6 | .906  | .853  | .869 | .036 | .899 | .915  |   
CPD     | 29.2    | 85.9   | 10.5     | 36.3 | .886  | .815  | .792 | .052 | .885 | .888  |   
PoolNet | 52.5    | 236.2  | 26.4     | 23.1 | .902  | .850  | .852 | .039 | .898 | .913  |   
EGNet   | 101     | 178.8  | 19.2     | 16.3 | .909  | .853  | .859 | .037 | .904 | .914  |   
SCRN    | 16.3    | 47.2   | 9.3      | 24.8 | .896  | .820  | .822 | .046 | .891 | .894  |    
GCPA    | 42.8    | 197.1  | 17.5     | 29.3 | .903  | .836  | .845 | .041 | .898 | .907  |    
ITSD    | 16.9    | 76.3   | 15.2*    | 30.6 | .905  | .820  | .834 | .045 | .901 | .896  |  
MINet   | 47.8    | 162    | 21.8     | 23.4 | .900  | .839  | .852 | .039 | .895 | .909  |    


### ResNet50-based:
Methods | #Param. | GFLOPs | Tr. Time | FPS  | max-F | ave-F | Fbw  | MAE  | SM   | EM    | Weight
 ----   | ---     | -----  | -----    | ---- | ----- | ----- | ---- | ---- | ---- | ----- | ----     
DHSNet  | 24.2    | 13.8   | 3.9      | 49.2 | .909  | .830  | .848 | .039 | .905 | .905  |      
Amulet  | 79.8    | 1093.8 | 6.3      | 35.1 | .895  | .822  | .835 | .042 | .894 | .900  |    
NLDF    | 41.1    | 115.1  | 9.2      | 30.5 | .903  | .837  | .855 | .038 | .898 | .910  |   
SRM     | 61.2    | 20.2   | 5.5      | 34.3 | .882  | .803  | .812 | .047 | .885 | .891  |   
PicaNet | 106.1   | 36.9   | 18.5*    | 14.8 | .904  | .823  | .843 | .041 | .902 | .902  |  
DSS     | 134.3   | 35.3   | 6.6      | 27.3 | .894  | .821  | .826 | .045 | .893 | .898  |    
BASNet  | 95.5    | 47.2   | 12.2     | 32.8 | .917  | .861  | .884 | .032 | .909 | .921  |   
CPD     | 47.9    | 14.7   | 7.7      | 22.7 | .906  | .842  | .836 | .040 | .904 | .908  |   
PoolNet | 68.3    | 66.9   | 10.2     | 33.9 | .912  | .843  | .861 | .036 | .907 | .912  |   
EGNet   | 111.7   | 222.8  | 25.7     | 10.2 | .917  | .851  | .867 | .036 | .912 | .914  |   
SCRN    | 25.2    | 12.5   | 5.5      | 19.3 | .910  | .838  | .845 | .040 | .906 | .905  |    
GCPA    | 67.1    | 54.3   | 6.8      | 37.8 | .916  | .841  | .866 | .035 | .912 | .912  |    
ITSD    | 25.7    | 19.6   | 5.7      | 29.4 | .913  | .825  | .842 | .042 | .907 | .899  |  
MINet   | 162.4   | 87     | 11.7     | 23.5 | .913  | .851  | .871 | .034 | .906 | .917  |    


## Create New Model

If your want to create a new model, you can copy the template folder and modify it as you want.
```
cp -r ./methods/template ./methods/new_name
```
More details please refer to python files in template floder.

## Loss Factory

We supply a **Loss Factory** for an easier way to tune the loss function.
You can set --loss and --lw parameters to use it.

Here are some examples:
```
loss_dict = {'b': BCE, 's': SSIM, 'i': IOU, 'd': DICE, 'e': Edge, 'c': CTLoss}

python train.py ... --loss=bd
# loss = 1 * bce_loss + 1 * dice_loss

python train.py ... --loss=bs --lw=0.3,0.7
# loss = 0.3 * bce_loss + 0.7 * ssim_loss

python train.py ... --loss=bsid --lw=0.3,0.1,0.5,0.2
# loss = 0.3 * bce_loss + 0.1 * ssim_loss + 0.5 * iou_loss + 0.2 * dice_loss
```
