# SALOD

Source code of our work: "[Benchmarking Deep Models for Salient Object Detection](https://arxiv.org/abs/2202.02925)".   
In this works, we propose a new SALient Object Detection (SALOD) benchmark.

We re-implement 20 SOD methods using the same settings, including input size, data loader and evaluation metrics (thanks to [Metrics](https://github.com/lartpang/Py-SOD-VOS-EvalToolkit)). Hyperparameters of optimizer are different because of various network structures and objective functions. We try our best to tune the optimizer for these models to achieve the best performance one-by-one. Some other networks are debugging now, it is welcome for your contributions on these models.

**You can contact me through the official email: zhouhj26@mail2.sysu.edu.cn**  

## Trend
Here we show the performance trend of the ave-F score on ECSSD dataset.  
Results of PICANet, ITSD, EGNet and EDN will coming soon.   
All models are trained with the following setting:  
1. ```--strategy=sche_f3net``` for the latest training strategy as original F3Net, LDF, PFSNet and CTDNet;
2. ```--multi``` for multi-scale training;
3. ```--data_aug``` for random croping;  
4. 1 * BCE_loss + 1 * IOU_loss as loss.

![Result](https://github.com/moothes/SALOD/blob/master/trend.png)


# Update Log
## 2022/08/09  
Update the trend figure.  
New model: SCFNet (ECCV 2022).

## 2022/06/14  
Add performance trend figure.  
New model: EDN (TIP 2022).

## 2022/05/25  
**Fix a bug in evaluation.**  
In the previous versions, we found that images with large salient regions get 0 ave-F scores, and thus we obtain lower ave-F scores than their original paper.    
Now, we fix this bug by adding a round function before evaluating.

## 2022/05/15 
1. New models: F3Net (AAAI 2020), LDF (CVPR 2020), GateNet (ECCV 2020), PFSNet (AAAI 20221), CTDNet (ACM MM 2021). More models for SOD and COD tasks are coming soon.  
2. New dataset: training on COD task is available now.
3. Training strategy update. We notice that training strategy is very important for achieving SOTA performance. A new strategy factory is added to /base/strategy.py.


## Available Methods:

 Methods | Publish. | Paper | Src Code
 ----    | -----    | ----- | ------ 
 SCFNet | ECCV 2022 | [Arxiv](https://arxiv.org/abs/2208.02178)|  [Paddle](https://github.com/zhangjinCV/KD-SCFNet)
 EDN  | TIP 2022 | [TIP](https://ieeexplore.ieee.org/abstract/document/9756227/)|  [Pytorch](https://github.com/yuhuan-wu/EDN)
 CTDNet  | ACM MM 2021 | [ACM](https://dl.acm.org/doi/abs/10.1145/3474085.3475494?casa_token=eKn8q7l2hJEAAAAA%3A4YGBXBpC6cCcFdpekxbaZncgBEru_mi69kNixfZSPeFRhD2gkeKpXIZyuiIW1bH80IuNV9ANmBw)|  [Pytorch](https://github.com/zhaozhirui/CTDNet)
 PFSNet  | AAAI 2021 | [AAAI.org](https://ojs.aaai.org/index.php/AAAI/article/view/16331)|  [Pytorch](https://github.com/iCVTEAM/PFSNet)
 GateNet | ECCV 2020 | [springer](https://link.springer.com/chapter/10.1007/978-3-030-58536-5_3)|  [Pytorch](https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency)
 LDF     | CVPR 2020 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Label_Decoupling_Framework_for_Salient_Object_Detection_CVPR_2020_paper.html)|  [Pytorch](https://github.com/weijun88/LDF)
 MINet   | CVPR 2020 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/lartpang/MINet)  
 ITSD    | CVPR 2020 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/moothes/ITSD-pytorch)  
 GCPA    | AAAI 2020 | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6633) | [Pytorch](https://github.com/JosephChenHub/GCPANet)  
 F3Net   | AAAI 2020 | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6916) | [Pytorch](https://github.com/weijun88/F3Net)  
 SCRN    | ICCV 2019 | [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf) | [Pytorch](https://github.com/wuzhe71/SCRN)  
 EGNet   | ICCV 2019 | [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf) | [Pytorch](https://github.com/JXingZhao/EGNet)  
 PoolNet | CVPR 2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_A_Simple_Pooling-Based_Design_for_Real-Time_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/backseason/PoolNet)  
 CPD     | CVPR 2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Cascaded_Partial_Decoder_for_Fast_and_Accurate_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/wuzhe71/CPD)  
 BASNet  | CVPR 2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/NathanUA/BASNet)  
 DSS     | TPAMI 2019| [IEEE](https://ieeexplore.ieee.org/document/8315520/)/[ArXiv](https://arxiv.org/abs/1611.04849) | [Pytorch](https://github.com/AceCoooool/DSS-pytorch)  
 PicaNet | CVPR 2018 | [openaccess](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_PiCANet_Learning_Pixel-Wise_CVPR_2018_paper.pdf) | [Pytorch](https://github.com/Ugness/PiCANet-Implementation)  
 SRM     | ICCV 2017 | [openaccess](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_A_Stagewise_Refinement_ICCV_2017_paper.pdf) | [Pytorch](https://github.com/xsxszab/SRM-Pytorch) 
 Amulet  | ICCV 2017 | [openaccess](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Amulet_Aggregating_Multi-Level_ICCV_2017_paper.pdf) | [Pytorch](https://github.com/xsxszab/Amulet-Pytorch)  
 NLDF    | CVPR 2017 | [openaccess](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Non-Local_Deep_Features_CVPR_2017_paper.pdf) | [Pytorch](https://github.com/AceCoooool/NLDF-pytorch)/[TF](https://github.com/zhimingluo/NLDF) 
 DHSNet  | CVPR 2016 | [openaccess](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf) | [Pytorch](https://github.com/xsxszab/DHSNet-Pytorch)  
 `Tuning`  | -----   | ----- | -----
 *PAGE    | CVPR2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Salient_Object_Detection_With_Pyramid_Attention_and_Salient_Edges_CVPR_2019_paper.pdf) | [TF](https://github.com/wenguanwang/PAGE-Net)  
 *PFA     | CVPR2019 | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/dizaiyoufang/pytorch_PFAN) 
 *PFPN   | AAAI2020 | [aaai.org](https://ojs.aaai.org/index.php/AAAI/article/view/6892) | [Pytorch](https://github.com/chenquan-cq/PFPN)
 
 

## Datasets
Our SALOD dataset can be downloaded from: [SALOD](https://drive.google.com/file/d/1kxhUoWUAnFhOE_ZoA1www8msG2pKHg3_/view?usp=sharing).   
Original SOD datasets from: [SOD](https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view?usp=sharing), including DUTS-TR,DUTS-TE,ECSSD,SOD,PASCAL-S,HKU-IS,DUT-OMRON.  
COD datasets from: [COD](https://drive.google.com/file/d/1zUgaGxr9PeDcfLfBisV2q8QXL6Tp1QzC/view?usp=sharing), including COD-TR (COD-TR + CAMO-TR), COD-TE, CAMO-TE, NC4K.

We have no plan on providing Baidu Disk links.
For chinese users who cannot open Google, I recommend you to purchase an SSR service in [Airport](https://52bp.org/airport.html).

## Properties
1. **A unified interface for new models.** To develop a new model, you only need to 1) set configs; 2) define network; 3) define loss function. See methods/template.
2. Setting different backbones through ```--backbone```. **(Available backbones: ResNet-50, VGG-16, MobileNet-v2, EfficientNet-B0, GhostNet, Res2Net)[[Weight]](https://drive.google.com/drive/folders/1Rxo2e38Tj_xUtLhCa_04S1YnYtWaEYgs?usp=sharing)**
3. **Testing all models on your own device.** You can test all available methods in our benchmark, including FPS, MACs, model size and multiple effectiveness metrics.
4. We implement a **loss factory** that you can change the loss functions through ```--loss``` and ```--lw```.

 
 ## Usage
 
 ```
 # model_name: lower-cased method name. E.g. poolnet, egnet, gcpa, dhsnet or minet.
 python3 train.py model_name --gpus=0 --trset=[DUTS-TR,SALOD,COD-TR]
 
 python3 test.py model_name --gpus=0 --weight=path_to_weight [--save]
 
 python3 test_fps.py model_name --gpus=0
 
 # To evaluate generated maps:
 python3 eval.py --pre_path=path_to_maps
 ```
 
 
 
 ## Results
 
 We report benchmark results here.  
 More results please refer to [Reproduction](https://github.com/moothes/SALOD/blob/master/readme/Reproduction.md), [Few-shot](https://github.com/moothes/SALOD/blob/master/readme/Few-shot.md) and [Generalization](https://github.com/moothes/SALOD/blob/master/readme/Generazation.md).
 
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

To create a new model, you can copy the template folder and modify it as you want.
```
cp -r ./methods/template ./methods/new_name
```
More details please refer to python files in template folder.

## Loss Factory

We supply a **Loss Factory** for an easier way to tune the loss functions.
You can set --loss and --lw parameters to use it.

Here are some examples:
```
loss_dict = {'b': BCE, 's': SSIM, 'i': IOU, 'd': DICE, 'e': Edge, 'c': CTLoss}

python train.py basnet --loss=bd
# loss = 1 * bce_loss + 1 * dice_loss

python train.py basnet --loss=bs --lw=0.3,0.7
# loss = 0.3 * bce_loss + 0.7 * ssim_loss

python train.py basnet --loss=bsid --lw=0.3,0.1,0.5,0.2
# loss = 0.3 * bce_loss + 0.1 * ssim_loss + 0.5 * iou_loss + 0.2 * dice_loss
```

Thanks for citing our work
```xml
@article{salod,
  title={Benchmarking Deep Models for Salient Object Detection},
  author={Zhou, Huajun and Lin, Yang and Yang, Lingxiao and Lai, Jianhuang and Xie, Xiaohua},
  journal={arXiv preprint arXiv:2202.02925},
  year={2022}
}
```
