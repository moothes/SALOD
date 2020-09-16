# SOD_hub

This project is a new benchmark for existing saliency object detection models. 

In order to make the comparison as fair as possible, we use same settings for all networks, including input size, data loader and evaluation metrics (thanks to [Metrics](https://github.com/lartpang/Py-SOD-VOS-EvalToolkit)). Training strategies are different because of various network structures and objective functions. We try our best to tune the optimer for these models to achieve best performance one by one. 

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
 *PAGE    | CVPR2019 | 320^2 | ------ | ------ | ----- | ----- | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Salient_Object_Detection_With_Pyramid_Attention_and_Salient_Edges_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/wenguanwang/PAGE-Net)  
 *PFA     | CVPR2019 | 320^2 | ------ | ------ | ----- | ----- | ----- | [openaccess](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Pyramid_Feature_Attention_Network_for_Saliency_Detection_CVPR_2019_paper.pdf) | [Pytorch](https://github.com/dizaiyoufang/pytorch_PFAN)  
 *F3Net   | AAAI2020 | 320^2 | ------ | ------ | ----- | ----- | ----- | [aaai.org](https://aaai.org/ojs/index.php/AAAI/article/view/6916) | [Pytorch](https://github.com/weijun88/F3Net)  

