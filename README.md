
# SALOD
Welcome to The Alchemist's Hut. Here is the SALOD benchmark ([paper link](https://arxiv.org/abs/2202.02925)) for the SOD and COD (in the near future) tasks.  

We have re-implemented over 20 SOD methods using the same settings, including input size, data loader and evaluation metrics (thanks to [Metrics](https://github.com/lartpang/Py-SOD-VOS-EvalToolkit)). Some other networks are debugging now, it is welcome for your contributions on these models.

Available backbones: (ResNet-50, VGG-16, MobileNet-v2, EfficientNet-B0, GhostNet, Res2Net)[[Weights]](https://drive.google.com/drive/folders/1Rxo2e38Tj_xUtLhCa_04S1YnYtWaEYgs?usp=sharing)  

*Our new unsupervised SOD method [A2S-v2 framework](https://github.com/moothes/A2S-v2) is public available now!*

**You can contact me through the official email: zhouhj26@mail2.sysu.edu.cn**  

## Datasets
Our SALOD dataset can be downloaded from: [SALOD](https://drive.google.com/file/d/1kxhUoWUAnFhOE_ZoA1www8msG2pKHg3_/view?usp=sharing).   
Original SOD datasets from: [SOD](https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view?usp=sharing), including DUTS-TR,DUTS-TE,ECSSD,SOD,PASCAL-S,HKU-IS,DUT-OMRON.  
COD datasets from: [COD](https://drive.google.com/file/d/1zUgaGxr9PeDcfLfBisV2q8QXL6Tp1QzC/view?usp=sharing), including COD-TR (COD-TR + CAMO-TR), COD-TE, CAMO-TE, NC4K.
 
## Results
Here we show the performance trend of the ave-F score on ECSSD dataset.   
The weights of these models can be downloaded from: [Baidu Disk](https://pan.baidu.com/s/1ByHuao32_2fUSXV7nNNMIA)(cs6u)  
Results of PICANet and EDN will coming soon.   
All models are trained with the following setting:  
1. ```--strategy=sche_f3net``` for the latest training strategy as original F3Net, LDF, PFSNet and CTDNet;
2. ```--multi``` for multi-scale training;
3. ```--data_aug``` for random croping;  
4. 1 * BCE_loss + 1 * IOU_loss as loss.

### ResNet50-based:
Methods | #Para. | GMACs  | FPS  | PASCAL-S |   -  | ECSSD  |  -   | HKU-IS |  -   | DUTS-TE |  -   | DUT-OMRON |    -   
 ----   | ---    | -----  | ---- | ----- | ----- | ---- | ---- | ---- | ----- | ---- | -----| ---- | -----            
Methods | #Para. | GMACs  | FPS  | ave-F | MAE   | ave-F| MAE  | ave-F| MAE   | ave-F| MAE  | ave-F| MAE   
DHSNet  | 24.2   | 13.8   | 49.2 | .822  | .064  | .919 | .036 | .902 | .031  | .826 | .039 | .756 | .056      
Amulet  | 79.8   | 1093.8 | 35.1 | .816  | .070  | .911 | .041 | .895 | .034  | .813 | .042 | .741 | .058 
NLDF    | 41.1   | 115.1  | 30.5 | .821  | .064  | .916 | .036 | .898 | .032  | .819 | .040 | .745 | .060 
SRM     | 61.2   | 20.2   | 34.3 | .809  | .072  | .898 | .045 | .877 | .040  | .794 | .047 | .731 | .062  
DSS     | 134.3  | 35.3   | 27.3 | .790  | .085  | .889 | .050 | .877 | .041  | .766 | .054 | .729 | .064 
BASNet  | 95.5   | 47.2   | 32.8 | .818  | .072  | .914 | .037 | .908 | .031  | .832 | .043 | .774 | .058  
CPD     | 47.9   | 14.7   | 22.7 | .837  | .062  | .923 | .035 | .909 | .031  | .841 | .037 | .776 | .053  
PoolNet | 68.3   | 66.9   | 33.9 | .833  | .062  | .924 | .033 | .906 | .030  | .836 | .037 | .765 | .057 
EGNet   | 111.7  | 222.8  | 10.2 | .828  | .063  | .917 | .036 | .902 | .031  | .836 | .039 | .762 | .059  
SCRN    | 25.2   | 12.5   | 19.3 | .835  | .061  | .923 | .034 | .907 | .031  | .842 | .037 | .771 | .057  
F3Net   | 25.5   | 13.6   | 39.2 | .819  | .068  | .917 | .038 | .903 | .032  | .821 | .042 | .757 | .058  
GCPA    | 67.1   | 54.3   | 37.8 | .832  | .066  | .925 | .033 | .910 | .030  | .841 | .040 | .773 | .059  
ITSD    | 25.7   | 19.6   | 29.4 | .812  | .073  | .913 | .039 | .902 | .033  | .820 | .048 | .765 | .065  
MINet   | 162.4  | 87     | 23.5 | .834  | .062  | .924 | .035 | .909 | .029  | .843 | .037 | .769 | .054  
LDF     | 25.2   | 12.8   | 37.5 | .831  | .061  | .921 | .035 | .903 | .030  | .835 | .038 | .763 | .056  
GateNet | 128.6  | 96     | 25.9 | .825  | .069  | .920 | .036 | .908 | .031  | .834 | .040 | .770 | .057  
PFSNet  | 31.2   | 37.5   | 21.7 | .838  | .060  | .926 | .033 | .910 | .029  | .845 | .036 | .770 | .055  
CTDNet  | 24.6   | 10.2   | 64.2 | .830  | .065  | .922 | .035 | .905 | .030  | .833 | .039 | .773 | .054  

![Result](https://github.com/moothes/SALOD/blob/master/trend.png)

## Benchmarking results
[weights](https://pan.baidu.com/s/1KXFU09nBElHqP9ffdHWtNw) code: pqn6


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
 
 

 ## Usage
 
 ```
 # model_name: lower-cased method name. E.g. poolnet, egnet, gcpa, dhsnet or minet.
 python3 train.py model_name --gpus=0 --trset=[DUTS-TR,SALOD,COD-TR]
 
 python3 test.py model_name --gpus=0 --weight=path_to_weight [--save]
 
 python3 test_fps.py model_name --gpus=0
 
 # To evaluate generated maps:
 python3 eval.py --pre_path=path_to_maps
 ```
 
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

# Update Log
2022/10/17:  
* Use ```timm``` library for more backbones.
* Code update.

2022/08/09:  
* Remove loss.py for each method. The loss functions are defined in config.py now.  
* Weights are uploaded to Baidu Disk.

2022/08/09:  
* Update the trend figure.  
* New model: SCFNet (ECCV 2022).

2022/06/14: 
* Add performance trend figure.  
* New model: EDN (TIP 2022).

2022/05/25:    
* In the previous versions, we found that images with large salient regions get 0 ave-F scores, and thus we obtain lower ave-F scores than their original paper. Now, we fix this bug by adding a round function before evaluating.

2022/05/15: 
* New models: F3Net (AAAI 2020), LDF (CVPR 2020), GateNet (ECCV 2020), PFSNet (AAAI 20221), CTDNet (ACM MM 2021). More models for SOD and COD tasks are coming soon. 
* New dataset: training on COD task is available now.
* Training strategy update. We notice that training strategy is very important for achieving SOTA performance. A new strategy factory is added to /base/strategy.py.

Thanks for citing our work
```xml
@article{salod,
  title={Benchmarking Deep Models for Salient Object Detection},
  author={Zhou, Huajun and Lin, Yang and Yang, Lingxiao and Lai, Jianhuang and Xie, Xiaohua},
  journal={arXiv preprint arXiv:2202.02925},
  year={2022}
}
```
