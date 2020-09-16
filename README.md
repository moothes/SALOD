# SOD_hub

This project is a new benchmark for existing saliency object detection models. 

In order to make the comparison as fair as possible, we use same settings for all networks, including input size, data loader and evaluation metrics (thanks to [Metrics](https://github.com/lartpang/Py-SOD-VOS-EvalToolkit)). Training strategies are different because of various network structures and objective functions. We try our best to tune the optimer for these models to achieve best performance one by one. 

There are 14 networks from top conferences (CVPR, ICCV, AAAI) or top journals (TPAMI) these years are available now. Notice that only the network achieve its performance as reported in their paper, it will be available in this project. 

## Methods:

 Methods | Publish. | Input | Param. | Optim. | LR    | Epoch | Time  | Paper | Src Code
 ----    | -----    | ----- | ------ | ------ | ----- | ----- | ----- | ----- | ------
 DHSNet  | CVPR2016 | 320^2 | 95M    | Adam   | 2e-5  | 30    | ----- | ----- | [Pytorch](https://github.com/xsxszab/DHSNet-Pytorch)  
 Amulet  | ICCV2017 | 320^2 | 312M   | Adam   | 1e-5  | 30    | ----- | ----- | [Pytorch](https://github.com/xsxszab/Amulet-Pytorch)  
 NLDF    | CVPR2017 | 320^2 | 161M   | Adam   | 1e-5  | 30    | ----- | ----- | [Pytorch](https://github.com/AceCoooool/NLDF-pytorch)/[TF](https://github.com/zhimingluo/NLDF) 
 SRM     | ICCV2017 | 320^2 | 240M   | Adam   | 5e-5  | 30    | ----- | ----- | [Pytorch](https://github.com/xsxszab/SRM-Pytorch) 
 PicaNet | CVPR2018 | 320^2 | 464M   | SGD    | 1e-2  | 30    | ----- | ----- | [Pytorch](https://github.com/Ugness/PiCANet-Implementation)  
 DSS     | TPAMI2019| 320^2 | 525M   | Adam   | 2e-5  | 30    | ----- | ----- | [Pytorch](https://github.com/AceCoooool/DSS-pytorch)  
 BASNet  | CVPR2019 | 320^2 | 374M   | Adam   | 1e-5  | 25    | ----- | ----- | [Pytorch](https://github.com/NathanUA/BASNet)  
 CPD     | CVPR2019 | 320^2 | 188M   | Adam   | 1e-5  | 30    | ----- | ----- | [Pytorch](https://github.com/wuzhe71/CPD)  
 PoolNet | CVPR2019 | 320^2 | 267M   | Adam   | 5e-5  | 30    | ----- | ----- | [Pytorch](https://github.com/backseason/PoolNet)  
 EGNet   | ICCV2019 | 320^2 | 437M   | Adam   | 5e-5  | 30    | ----- | ----- | [Pytorch](https://github.com/JXingZhao/EGNet)  
 SCRN    | ICCV2019 | 320^2 | 100M   | SGD    | 1e-2  | 30    | ----- | ----- | [Pytorch](https://github.com/wuzhe71/SCRN)  
 GCPA    | AAAI2020 | 320^2 | 263M   | SGD    | 1e-2  | 30    | ----- | ----- | [Pytorch](https://github.com/JosephChenHub/GCPANet)  
 ITSD    | CVPR2020 | 320^2 | 101M   | SGD    | 5e-3  | 30    | ----- | ----- | [Pytorch](https://github.com/moothes/ITSD-pytorch)  
 MINet   | CVPR2020 | 320^2 | 635M   | SGD    | 1e-3  | 30    | ----- | ----- | [Pytorch](https://github.com/lartpang/MINet)  
 `Tuning`  | -----    | ----- | ------ | ------ | ----- | ----- | ----- | ----- | -----
 *PAGE    | CVPR2019 | 320^2 | ------ | ------ | ----- | ----- | ----- | ----- | [Pytorch](https://github.com/wenguanwang/PAGE-Net)  
 *PFA     | CVPR2019 | 320^2 | ------ | ------ | ----- | ----- | ----- | ----- | [Pytorch](https://github.com/dizaiyoufang/pytorch_PFAN)  
 *F3Net   | AAAI2020 | 320^2 | ------ | ------ | ----- | ----- | ----- | ----- | [Pytorch](https://github.com/weijun88/F3Net)  

