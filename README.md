# SALOD

A new benchmark for SALiency Object Detection (SALOD) task. 

In order to make the comparison as fair as possible, we use same settings for all networks, including input size, data loader and evaluation metrics (thanks to [Metrics](https://github.com/lartpang/Py-SOD-VOS-EvalToolkit)). Training strategies are different because of various network structures and objective functions. We try our best to tune the optimizer for these models to achieve best performance one by one. 

There are 14 networks from top conferences (CVPR, ICCV, AAAI) or top journals (TPAMI) these years are available now. Notice that only the network achieve comparable or better performance as reported in their paper, it will be available in this project. 

## Methods:

 Methods | Publish. | Input | Param. | Optim. | LR    | Epoch | Time  | Paper | Src Code
 ----    | -----    | ----- | ------ | ------ | ----- | ----- | ----- | ----- | ------
 DHSNet  | CVPR2016 | 320^2 | 95M    | Adam   | 2e-5  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf) | [Pytorch](https://github.com/xsxszab/DHSNet-Pytorch)  
 NLDF    | CVPR2017 | 320^2 | 161M   | Adam   | 1e-5  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Non-Local_Deep_Features_CVPR_2017_paper.pdf) | [Pytorch](https://github.com/AceCoooool/NLDF-pytorch)/[TF](https://github.com/zhimingluo/NLDF) 
 Amulet  | ICCV2017 | 320^2 | 312M   | Adam   | 1e-5  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Amulet_Aggregating_Multi-Level_ICCV_2017_paper.pdf) | [Pytorch](https://github.com/xsxszab/Amulet-Pytorch)  
 SRM     | ICCV2017 | 320^2 | 240M   | Adam   | 5e-5  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_A_Stagewise_Refinement_ICCV_2017_paper.pdf) | [Pytorch](https://github.com/xsxszab/SRM-Pytorch) 
 PicaNet | CVPR2018 | 320^2 | 464M   | SGD    | 1e-2  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_PiCANet_Learning_Pixel-Wise_CVPR_2018_paper.pdf) | [Pytorch](https://github.com/Ugness/PiCANet-Implementation)  
 DSS     | TPAMI2018| 320^2 | 525M   | Adam   | 2e-5  | 30    | ----- | [IEEE](https://ieeexplore.ieee.org/document/8315520/)/[ArXiv](https://arxiv.org/abs/1611.04849) | [Pytorch](https://github.com/AceCoooool/DSS-pytorch)  
 BASNet  | CVPR2019 | 320^2 | 374M   | Adam   | 1e-5  | 25    | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/NathanUA/BASNet)  
 CPD     | CVPR2019 | 320^2 | 188M   | Adam   | 1e-5  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Cascaded_Partial_Decoder_for_Fast_and_Accurate_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/wuzhe71/CPD)  
 PoolNet | CVPR2019 | 320^2 | 267M   | Adam   | 5e-5  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_A_Simple_Pooling-Based_Design_for_Real-Time_Salient_Object_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/backseason/PoolNet)  
 EGNet   | ICCV2019 | 320^2 | 437M   | Adam   | 5e-5  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_EGNet_Edge_Guidance_Network_for_Salient_Object_Detection_ICCV_2019_paper.pdf) | [Pytorch](https://github.com/JXingZhao/EGNet)  
 SCRN    | ICCV2019 | 320^2 | 100M   | SGD    | 1e-2  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.pdf) | [Pytorch](https://github.com/wuzhe71/SCRN)  
 GCPA    | AAAI2020 | 320^2 | 263M   | SGD    | 1e-2  | 30    | ----- | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6633) | [Pytorch](https://github.com/JosephChenHub/GCPANet)  
 ITSD    | CVPR2020 | 320^2 | 101M   | SGD    | 5e-3  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/moothes/ITSD-pytorch)  
 MINet   | CVPR2020 | 320^2 | 635M   | SGD    | 1e-3  | 30    | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Multi-Scale_Interactive_Network_for_Salient_Object_Detection_CVPR_2020_paper.pdf) | [Pytorch](https://github.com/lartpang/MINet)  
 `Tuning`  | -----    | ----- | ------ | ------ | ----- | ----- | ----- | ----- | -----
 *PAGE    | CVPR2019 | 320^2 | ------ | ------ | ----- | ----- | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Salient_Object_Detection_With_Pyramid_Attention_and_Salient_Edges_CVPR_2019_paper.pdf) | [TF](https://github.com/wenguanwang/PAGE-Net)  
 *PFA     | CVPR2019 | 320^2 | ------ | ------ | ----- | ----- | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/dizaiyoufang/pytorch_PFAN)  
 *F3Net   | AAAI2020 | 320^2 | ------ | ------ | ----- | ----- | ----- | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6916) | [Pytorch](https://github.com/weijun88/F3Net)  
 
 ## Usage
 
 To train a model:
 ```
 # model_name: lower-cased method name. E.g. poolnet, egnet, gcpa, dhsnet or minet.
 python train.py model_name --gpus=0
 ```
 
 To test a model:
 ```
 # model_name: lower-cased method name. E.g. poolnet, egnet, gcpa, dhsnet or minet.
 python test.py model_name --gpus=0 --weight=path_to_weight
 ```
 
 To evaluate generated maps:
 ```
 python eval.py --pre_path=path_to_maps
 ```
 
 
 
 ## Results
 
 We only report max-F score here. You can [download] the trained models or official maps to get more results. 
 
 Src indicates saliency maps from official code, while ss and ms mean single-scale and multi-scale training respectively.
 
 **Notice: If you have a new optimizer strategy that can train any network to get a higher score, we would appreciate it.**
 
 
Methods | Source | Backbone  | SOD   | Pascal-S | ECSSD | HKU-IS | DUTS-TE | DUT-OMRON 
 ----   | ---    | -----     | ----- | -------- | ----- | -----  | -----   | -----     
DHSNet  | src    | VGG16     | .827  | .820     | .906  | .890   | .808    | ------    
----    | ss     | Resnet50  | .868  | .865     | .940  | .930   | .870    | .796      
----    | ms     | Resnet50  | .873  | .864     | .943  | .930   | .880    | .807      
Amulet  | src    | VGG16     | .798  | .828     | .915  | .897   | .778    | .743      
----    | ss     | Resnet50  | .868  | .856     | .933  | .924   | .860    | .781      
----    | ms     | Resnet50  | .861  | .861     | .933  | .925   | .867    | .788      
NLDF    | src    | VGG16     | .841  | .822     | .905  | .902   | .813    | .753      
----    | ss     | Resnet50  | .866  | .861     | .933  | .923   | .867    | .792      
----    | ms     | Resnet50  | .865  | .869     | .934  | .924   | .873    | .795      
SRM     | src    | Resnet50  | .843  | .838     | .917  | .906   | .826    | .769      
----    | ss     | Resnet50  | .841  | .835     | .919  | .901   | .824    | .763      
----    | ms     | Resnet50  | .847  | .840     | .922  | .904   | .832    | .773      
PicaNet | src    | Resnet50  | .856  | .857     | .935  | .918   | .860    | .803      
----    | ss     | Resnet50  | .867  | .857     | .934  | .923   | .869    | .797      
----    | ms     | Resnet50  | .860  | .857     | .933  | .917   | .867    | .800      
DSS     | src    | Resnet50  | .846  | .831     | .921  | .900   | .826    | .769      
----    | ss     | Resnet50  | .857  | .852     | .929  | .916   | .855    | .784      
----    | ms     | Resnet50  | .858  | .853     | .927  | .914   | .858    | .785      
BasNet  | src    | Resnet34  | .851  | .854     | .942  | .928   | .859    | .805      
----    | ss     | Resnet50  | .874  | .866     | .949  | .937   | .889    | .818      
----    | ms     | Resnet50  | .884  | .869     | .951  | .938   | .894    | .821      
CPD     | src    | Resnet50  | .860  | .859     | .939  | .925   | .865    | .797      
----    | ss     | Resnet50  | .860  | .867     | .937  | .925   | .871    | .798      
----    | ms     | Resnet50  | .866  | .871     | .941  | .928   | .876    | .809      
PoolNet | src    | Resnet50  | .871  | .863     | .944  | --     | .880    | .808      
----    | ss     | Resnet50  | .865  | .867     | .939  | .931   | .877    | .794      
----    | ms     | Resnet50  | .869  | .862     | .943  | .928   | .877    | .806      
EGNet   | src    | Resnet50  | .880  | .865     | .947  | --     | .889    | .815      
----    | ss     | Resnet50  | .870  | .863     | .946  | .930   | .879    | .811      
----    | ms     | Resnet50  | .871  | .869     | .948  | .931   | .887    | .817      
SCRN    | src    | Resnet50  | --    | .877     | .950  | .934   | .888    | .811      
----    | ss     | Resnet50  | .878  | .869     | .943  | .932   | .881    | .807      
----    | ms     | Resnet50  | .875  | .873     | .944  | .932   | .886    | .812      
GCPA    | src    | Resnet50  | .876  | .869     | .948  | .938   | .888    | .812      
----    | ss     | Resnet50  | .864  | .869     | .945  | .933   | .886    | .801      
----    | ms     | Resnet50  | .867  | .874     | .945  | .936   | .892    | .812      
ITSD    | src    | Resnet50  | .876  | .872     | .946  | .935   | .885    | .821      
----    | ss     | Resnet50  | .872  | .866     | .941  | .930   | .879    | .809      
----    | ms     | Resnet50  | .867  | .871     | .945  | .931   | .885    | .817      
MINet   | src    | Resnet50  | --    | .867     | .947  | .935   | .884    | .810      
----    | ss     | Resnet50  | .867  | .873     | .940  | .929   | .881    | .802      
----    | ms     | Resnet50  | .871  | .874     | .945  | .935   | .890    | .819      

## New Model

If your want to create a new model, you can copy the template folder and modify it as you want.
```
cp -r ./methods/template ./methods/new_name
```
More details please refer to python files in template floder.

## Loss Factory

We supply a **Loss Factory** for an easier way to tuning the loss function.
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
