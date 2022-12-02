
# SALOD
Welcome to The Alchemist's Hut. Here is the SALOD benchmark ([paper link](https://arxiv.org/abs/2202.02925)) for the SOD and COD (in the near future) tasks.  

We have re-implemented over 20 SOD methods using the same settings, including input size, data loader and evaluation metrics (thanks to [Metrics](https://github.com/lartpang/Py-SOD-VOS-EvalToolkit)). Some other networks are debugging now, it is welcome for your contributions on these models.

*Our new unsupervised SOD method [A2S-v2 framework](https://github.com/moothes/A2S-v2) is public available now!*

**You can contact me through the official email: zhouhj26@mail2.sysu.edu.cn**  

## Datasets
Our SALOD dataset can be downloaded from: [SALOD](https://drive.google.com/file/d/1kxhUoWUAnFhOE_ZoA1www8msG2pKHg3_/view?usp=sharing).   
Original SOD datasets from: [SOD](https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view?usp=sharing), including DUTS-TR,DUTS-TE,ECSSD,SOD,PASCAL-S,HKU-IS,DUT-OMRON.  
COD datasets from: [COD](https://drive.google.com/file/d/1zUgaGxr9PeDcfLfBisV2q8QXL6Tp1QzC/view?usp=sharing), including COD-TR (COD-TR + CAMO-TR), COD-TE, CAMO-TE, NC4K.
 
## Results
All models are trained with the following setting:  
1. ```--strategy=sche_f3net``` for the latest training strategy as original F3Net, LDF, PFSNet and CTDNet;
2. ```--multi``` for multi-scale training;
3. ```--data_aug``` for random croping;  
4. 1 * BCE_loss + 1 * IOU_loss as loss.

## Benchmarking results
Following the above settings, we list the benchmark results here.
All weights can be downloaded from [Baidu disk](https://pan.baidu.com/s/1KXFU09nBElHqP9ffdHWtNw) [pqn6].  
**Noted that FPS is tested on our device with batch_size=1, you should test all methods and report the scores on your own device.**

Methods | #Para. | GMACs  | FPS  | max-F | ave-F | Fbw  | MAE  | SM   | em    
 ----   | ---    | -----  | ---- | ----- | ----- | ---- | ---- | ---- | -----  
DHSNet  | 24.2   | 13.8   | 49.2 | .909  | .871  | .863 | .037 | .905 | .925      
Amulet  | 79.8   | 1093.8 | 35.1 | .897  | .856  | .846 | .042 | .896 | .919   
NLDF    | 41.1   | 115.1  | 30.5 | .908  | .868  | .859 | .038 | .903 | .930  
SRM     | 61.2   | 20.2   | 34.3 | .893  | .851  | .841 | .042 | .892 | .925    
DSS     | 134.3  | 35.3   | 27.3 | .906  | .868  | .859 | .038 | .901 | .933 
PiCaNet | 106.1  | 36.9   | 14.8 | .900  | .864  | .852 | .043 | .896 | .924  
BASNet  | 95.5   | 47.2   | 32.8 | .911  | .872  | .863 | .040 | .905 | .925  
CPD     | 47.9   | 14.7   | 22.7 | .913  | .884  | .874 | .034 | .911 | .938 
PoolNet | 68.3   | 66.9   | 33.9 | .916  | .882  | .875 | .035 | .911 | .938  
EGNet   | 111.7  | 222.8  | 10.2 | .913  | .884  | .875 | .036 | .908 | .936  
SCRN    | 25.2   | 12.5   | 19.3 | .916  | .881  | .872 | .035 | .910 | .935 
F3Net   | 25.5   | 13.6   | 39.2 | .911  | .878  | .869 | .036 | .908 | .932  
GCPA    | 67.1   | 54.3   | 37.8 | .914  | .884  | .874 | .036 | .910 | .937   
ITSD    | 25.7   | 19.6   | 29.4 | .918  | .880  | .873 | .037 | .910 | .932   
MINet   | 162.4  | 87     | 23.5 | .912  | .874  | .866 | .038 | .908 | .931 
LDF     | 25.2   | 12.8   | 37.5 | .913  | .879  | .873 | .035 | .909 | .938  
GateNet | 128.6  | 96     | 25.9 | .912  | .882  | .870 | .037 | .906 | .934 
PFSNet  | 31.2   | 37.5   | 21.7 | .912  | .879  | .865 | .038 | .904 | .931 
CTDNet  | 24.6   | 10.2   | 64.2 | .918  | .887  | .880 | .033 | .913 | .940  
EDN     | 35.1   | 16.1   | 27.4 | .916  | .883  | .875 | .036 | .910 | .934  
SCFNet  | 26.1   | 10.9   | 73.7 | .914  | .893  | .886 | .032 | .909 | .945  


# Conventional SOD results
The weights of these models can be downloaded from: [Baidu Disk](https://pan.baidu.com/s/1ByHuao32_2fUSXV7nNNMIA)(cs6u)    

<table>
 <tr>
<td width=60 rowspan=2>Method</td>
<td width=100 colspan=2>PASCAL-S</td>
<td width=100 colspan=2>ECSSD</td>
<td width=100 colspan=2>HKU-IS</td>
<td width=100 colspan=2>DUTS-TE</td>
<td width=100 colspan=2>DUT-OMRON</td>
 </tr>
 <tr>
<td>ave-F</td><td>MAE</td>
<td>ave-F</td><td>MAE</td>
<td>ave-F</td><td>MAE</td>
<td>ave-F</td><td>MAE</td>
<td>ave-F</td><td>MAE</td>
 </tr>
 <tr>
<td>DHSNet</td>
<td>.822</td><td>.064</td>
<td>.919</td><td>.036</td>
<td>.902</td><td>.031</td>
<td>.826</td><td>.039</td>
<td>.756</td><td>.056</td>
 </tr>
 <tr>
<td>Amulet</td>
<td>.816</td><td>.070</td>
<td>.911</td><td>.041</td>
<td>.895</td><td>.034</td>
<td>.813</td><td>.042</td>
<td>.741</td><td>.058</td>
 </tr>
 <tr>
<td>NLDF</td>
<td>.821</td><td>.064</td>
<td>.916</td><td>.036</td>
<td>.898</td><td>.032</td>
<td>.819</td><td>.040</td>
<td>.745</td><td>.060</td>
 </tr>
 <tr>
<td>SRM</td>
<td>.809</td><td>.072</td>
<td>.898</td><td>.045</td>
<td>.877</td><td>.040</td>
<td>.794</td><td>.047</td>
<td>.731</td><td>.062</td>
 </tr>
 <tr>
<td>DSS</td>
<td>.790</td><td>.085</td>
<td>.889</td><td>.050</td>
<td>.877</td><td>.041</td>
<td>.766</td><td>.054</td>
<td>.729</td><td>.064</td>
 </tr>
 <tr>
<td>BASNet</td>
<td>.818</td><td>.072</td>
<td>.914</td><td>.037</td>
<td>.908</td><td>.031</td>
<td>.832</td><td>.043</td>
<td>.774</td><td>.058</td>
 </tr>
 <tr>
<td>CPD</td>
<td>.837</td><td>.062</td>
<td>.923</td><td>.035</td>
<td>.909</td><td>.031</td>
<td>.841</td><td>.037</td>
<td>.776</td><td>.053</td>
 </tr>
 <tr>
<td>PoolNet</td>
<td>.833</td><td>.062</td>
<td>.924</td><td>.033</td>
<td>.906</td><td>.030</td>
<td>.836</td><td>.037</td>
<td>.765</td><td>.057</td>
 </tr>
 <tr>
<td>EGNet</td>
<td>.828</td><td>.063</td>
<td>.917</td><td>.036</td>
<td>.902</td><td>.031</td>
<td>.836</td><td>.039</td>
<td>.762</td><td>.059</td>
 </tr>
 <tr>
<td>SCRN</td>
<td>.835</td><td>.061</td>
<td>.923</td><td>.034</td>
<td>.907</td><td>.031</td>
<td>.842</td><td>.037</td>
<td>.771</td><td>.057</td>
 </tr>
 <tr>
<td>F3Net</td>
<td>.844</td><td>.055</td>
<td>.922</td><td>.032</td>
<td>.911</td><td>.029</td>
<td>.849</td><td>.035</td>
<td>.775</td><td>.056</td>
 </tr>
 <tr>
<td>GCPA</td>
<td>.832</td><td>.066</td>
<td>.925</td><td>.033</td>
<td>.910</td><td>.030</td>
<td>.841</td><td>.040</td>
<td>.773</td><td>.059</td>
 </tr>
 <tr>
<td>ITSD</td>
<td>.812</td><td>.073</td>
<td>.913</td><td>.039</td>
<td>.902</td><td>.033</td>
<td>.820</td><td>.048</td>
<td>.765</td><td>.065</td>
 </tr>
 <tr>
<td>MINet</td>
<td>.834</td><td>.062</td>
<td>.924</td><td>.035</td>
<td>.909</td><td>.029</td>
<td>.843</td><td>.037</td>
<td>.769</td><td>.054</td>
 </tr>
 <tr>
<td>LDF</td>
<td>.843</td><td>.056</td>
<td>.924</td><td>.032</td>
<td>.906</td><td>.029</td>
<td>.838</td><td>.035</td>
<td>.764</td><td>.056</td>
 </tr>
 <tr>
<td>GateNet</td>
<td>.825</td><td>.069</td>
<td>.920</td><td>.036</td>
<td>.908</td><td>.031</td>
<td>.834</td><td>.040</td>
<td>.770</td><td>.057</td>
 </tr>
 <tr>
<td>PFSNet</td>
<td>.838</td><td>.060</td>
<td>.926</td><td>.033</td>
<td>.910</td><td>.029</td>
<td>.845</td><td>.036</td>
<td>.770</td><td>.055</td>
 </tr>
 <tr>
<td>CTDNet</td>
<td>.849</td><td>.056</td>
<td>.925</td><td>.033</td>
<td>.910</td><td>.030</td>
<td>.852</td><td>.033</td>
<td>.775</td><td>.053</td>
 </tr>
 <tr>
<td>EDN</td>
<td>.848</td><td>.058</td>
<td>.924</td><td>.031</td>
<td>.917</td><td>.027</td>
<td>.860</td><td>.032</td>
<td>.793</td><td>.055</td>
 </tr>
</table>


## Available Methods:

 Methods | Publish. | Paper | Src Code
 ----    | -----    | ----- | ------ 
 SCFNet | ECCV 2022? | [Arxiv](https://arxiv.org/abs/2208.02178)|  [Paddle](https://github.com/zhangjinCV/KD-SCFNet)
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
* Benchmark results update.

2022/08/09:  
* Remove loss.py for each method. The loss functions are defined in config.py now.  
* Weights are uploaded to Baidu Disk.

2022/08/09:  
* Update the trend figure.  
* New model: SCFNet (ECCV 2022?).

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
